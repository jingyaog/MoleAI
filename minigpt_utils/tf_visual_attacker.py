import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from minigpt_tf_utils import tf_prompt_wrapper, tf_generator


def normalize(images):
    mean = tf.constant([0.48145466, 0.4578275, 0.40821073], dtype=tf.float32)
    std = tf.constant([0.26862954, 0.26130258, 0.27577711], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 3, 1, 1])
    std = tf.reshape(std, [1, 3, 1, 1])
    images = (images - mean) / std
    return images


def denormalize(images):
    mean = tf.constant([0.48145466, 0.4578275, 0.40821073], dtype=tf.float32)
    std = tf.constant([0.26862954, 0.26130258, 0.27577711], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 3, 1, 1])
    std = tf.reshape(std, [1, 3, 1, 1])
    images = images * std + mean
    return images


class TFAttacker:
    def __init__(self, args, model, targets, device='cuda:0', is_rtp=False):
        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp
        self.targets = targets
        self.num_targets = len(targets)
        self.loss_buffer = []
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def attack_unconstrained(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1/255):
        print('>>> batch_size:', batch_size)
        my_generator = tf_generator.Generator(model=self.model)
        import torch
        if isinstance(img, torch.Tensor):
            img_base = img.detach().cpu().numpy()
        else:
            img_base = img.numpy() if isinstance(img, tf.Tensor) else img
        img_base_tf = tf.constant(img_base, dtype=tf.float32)
        adv_noise = tf.Variable(tf.random.uniform(tf.shape(img_base_tf), dtype=tf.float32), trainable=True)
        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size
            with tf.GradientTape() as tape:
                tape.watch(adv_noise)
                x_adv_tf = normalize(adv_noise)
                x_adv = torch.from_numpy(x_adv_tf.numpy()).to(self.device)
                prompt = tf_prompt_wrapper.Prompt(
                    model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]], device=self.device
                )
                if len(prompt.img_embs) > 0 and len(prompt.img_embs[0]) > 0:
                    prompt.img_embs[0][0] = prompt.img_embs[0][0].repeat(batch_size, 1, 1)
                prompt.update_context_embs()
                target_loss = self.attack_loss(prompt, batch_targets)
                if isinstance(target_loss, torch.Tensor):
                    target_loss_tf = tf.constant(target_loss.detach().cpu().numpy(), dtype=tf.float32)
                else:
                    target_loss_tf = tf.constant(float(target_loss), dtype=tf.float32)
            grads = tape.gradient(target_loss_tf, adv_noise)
            if grads is not None:
                sign_grad = tf.sign(grads)
                adv_noise.assign(tf.clip_by_value(adv_noise - alpha * sign_grad, 0.0, 1.0))
            loss_val = float(target_loss.item() if isinstance(target_loss, torch.Tensor) else target_loss)
            self.loss_buffer.append(loss_val)
            print("target_loss: %f" % loss_val)
            if t % 20 == 0:
                self.plot_loss()
            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv_tf = normalize(adv_noise)
                x_adv = torch.from_numpy(x_adv_tf.numpy()).to(self.device)
                prompt.update_img_prompts([[x_adv]])
                if len(prompt.img_embs) > 0 and len(prompt.img_embs[0]) > 0:
                    prompt.img_embs[0][0] = prompt.img_embs[0][0].repeat(batch_size, 1, 1)
                prompt.update_context_embs()
                response, _ = my_generator.generate(prompt)
                print('>>>', response)
                adv_img_prompt = denormalize(adv_noise)
                self.save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))
        return denormalize(adv_noise)

    def attack_constrained(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1/255, epsilon=128/255):
        print('>>> batch_size:', batch_size)
        my_generator = tf_generator.Generator(model=self.model)
        import torch
        if isinstance(img, torch.Tensor):
            img_base = img.detach().cpu().numpy()
        else:
            img_base = img.numpy() if isinstance(img, tf.Tensor) else img
        img_base_tf = tf.constant(img_base, dtype=tf.float32)
        adv_noise = tf.Variable(
            tf.random.uniform(tf.shape(img_base_tf), minval=-epsilon, maxval=epsilon, dtype=tf.float32),
            trainable=True
        )
        adv_noise.assign(tf.clip_by_value(adv_noise + img_base_tf, 0.0, 1.0) - img_base_tf)
        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size
            with tf.GradientTape() as tape:
                tape.watch(adv_noise)
                x_adv_tf = img_base_tf + adv_noise
                x_adv_tf_norm = normalize(x_adv_tf)
                x_adv = torch.from_numpy(x_adv_tf_norm.numpy()).to(self.device)
                prompt = tf_prompt_wrapper.Prompt(
                    model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]], device=self.device
                )
                if len(prompt.img_embs) > 0 and len(prompt.img_embs[0]) > 0:
                    prompt.img_embs[0][0] = prompt.img_embs[0][0].repeat(batch_size, 1, 1)
                prompt.update_context_embs()
                target_loss = self.attack_loss(prompt, batch_targets)
                if isinstance(target_loss, torch.Tensor):
                    target_loss_tf = tf.constant(target_loss.detach().cpu().numpy(), dtype=tf.float32)
                else:
                    target_loss_tf = tf.constant(float(target_loss), dtype=tf.float32)
            grads = tape.gradient(target_loss_tf, adv_noise)
            if grads is not None:
                sign_grad = tf.sign(grads)
                adv_noise.assign(tf.clip_by_value(adv_noise - alpha * sign_grad, -epsilon, epsilon))
                adv_noise.assign(tf.clip_by_value(adv_noise + img_base_tf, 0.0, 1.0) - img_base_tf)
            loss_val = float(target_loss.item() if isinstance(target_loss, torch.Tensor) else target_loss)
            self.loss_buffer.append(loss_val)
            print("target_loss: %f" % loss_val)
            if t % 20 == 0:
                self.plot_loss()
            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv_tf = img_base_tf + adv_noise
                x_adv_tf_norm = normalize(x_adv_tf)
                x_adv = torch.from_numpy(x_adv_tf_norm.numpy()).to(self.device)
                prompt.update_img_prompts([[x_adv]])
                if len(prompt.img_embs) > 0 and len(prompt.img_embs[0]) > 0:
                    prompt.img_embs[0][0] = prompt.img_embs[0][0].repeat(batch_size, 1, 1)
                prompt.update_context_embs()
                response, _ = my_generator.generate(prompt)
                print('>>>', response)
                adv_img_prompt = denormalize(x_adv_tf)
                self.save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))
        return denormalize(img_base_tf + adv_noise)

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer)
        x_ticks = list(range(0, num_iters))
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()
        np.save('%s/loss.npy' % (self.args.save_dir), self.loss_buffer)

    def save_image(self, img_tensor, path):
        img_np = img_tensor.numpy() if isinstance(img_tensor, tf.Tensor) else img_tensor
        if len(img_np.shape) == 4:
            img_np = img_np[0]
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        from PIL import Image
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save(path)

    def attack_loss(self, prompts, targets):
        import torch
        context_embs = prompts.context_embs
        if len(context_embs) == 1:
            context_embs = context_embs * len(targets)
        assert len(context_embs) == len(targets), f"Unmatched batch size {len(context_embs)} != {len(targets)}"
        batch_size = len(targets)
        self.model.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.model.llama_tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        bos = torch.ones([1, 1], dtype=to_regress_tokens.input_ids.dtype, device=self.device) * self.model.llama_tokenizer.bos_token_id
        bos_embs = self.model.llama_model.model.embed_tokens(bos)
        pad = torch.ones([1, 1], dtype=to_regress_tokens.input_ids.dtype, device=self.device) * self.model.llama_tokenizer.pad_token_id
        pad_embs = self.model.llama_model.model.embed_tokens(pad)
        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )
        pos_padding = torch.argmin(T, dim=1)
        input_embs = []
        targets_mask = []
        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []
        for i in range(batch_size):
            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]
            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length])
            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length
            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)
        max_length = max(seq_tokens_length)
        attention_mask = []
        for i in range(batch_size):
            context_mask = torch.ones([1, context_tokens_length[i] + 1], dtype=torch.long).to(self.device).fill_(-100)
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = torch.ones([1, num_to_pad], dtype=torch.long).to(self.device).fill_(-100)
            targets_mask[i] = torch.cat([context_mask, targets_mask[i], padding_mask], dim=1)
            input_embs[i] = torch.cat([bos_embs, context_embs[i], input_embs[i], pad_embs.repeat(1, num_to_pad, 1)], dim=1)
            attention_mask.append(torch.LongTensor([[1] * (1 + seq_tokens_length[i]) + [0] * num_to_pad]))
        targets = torch.cat(targets_mask, dim=0).to(self.device)
        inputs_embs = torch.cat(input_embs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)
        outputs = self.model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return loss

