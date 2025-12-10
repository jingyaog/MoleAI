import tensorflow as tf
import numpy as np
from tqdm import tqdm 
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import save_img 


def normalize(images):
    mean = tf.constant([0.48145466, 0.4578275, 0.40821073],dtype = tf.float32)
    std = tf.constant([0.26862954, 0.26130258, 0.27577711],dtype = tf.float32)
    
    mean =tf.reshape(mean,[1,1,1,3])
    std = tf.reshape(std,[1,1,1,3])

    return (images - mean)/std

def denormalize(images):
    mean = tf.constant([0.48145466, 0.4578275, 0.40821073],dtype = tf.float32)
    std = tf.constant([0.26862954, 0.26130258, 0.27577711],dtype = tf.float32)
    
    mean =tf.reshape(mean,[1,1,1,3])
    std = tf.reshape(std,[1,1,1,3])

    return images * std + mean


class Attacker:

    def __init__(self, args, model, targets, device='cuda:0', is_rtp=False):

        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

    def attack_unconstrained(self, img, batch_size = 8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)

        adv_noise = tf.Variable(tf.random.uniform(tf.shape(img),0.0,1.0, dtype = img.dtype),trainable = True)
        

        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            x_adv = normalize(adv_noise).repeat(batch_size, 1, 1, 1)
            x_adv = tf.repeat(x_adv,batch_size,axis = 0)

            samples = {
                'image': x_adv,
                'text_input': [''] * batch_size,
                'text_output': batch_targets
            }

            with tf.GradientTape() as tape:
                tape.watch(adv_noise)

                model_output = self.model(samples)
                target_loss = model_output['loss']

            grads = tape.gradient(target_loss,adv_noise)
            adv_noise.assign(adv_noise - alpha*tf.sign(grads))    

            adv_noise.assign(tf.clip_by_value(adv_noise,0.0,1.0))

            self.loss_buffer.append(target_loss.numpy())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = normalize(adv_noise)


                print('>>> Sample Outputs')
                print(self.model.generate({"image": x_adv, "prompt": ''},
                                     use_nucleus_sampling=True, top_p=0.9, temperature=1))

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_img(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):

        print('>>> batch_size:', batch_size)

        adv_noise = tf.Variable(tf.random.uniform(tf.shape(img), -epsilon, epsilon, dtype =img.dtype), trainable = True)
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()


        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            x_adv = x + adv_noise
            x_adv = normalize(x_adv).repeat(batch_size, 1, 1, 1)

            samples = {
                'image': x_adv,
                'text_input': [''] * batch_size,
                'text_output': batch_targets
            }

            target_loss = self.model(samples)['loss']
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)

            
                print('>>> Sample Outputs')
                print(self.model.generate({"image": x_adv, "prompt": ''},
                                              use_nucleus_sampling=True, top_p=0.9, temperature=1))

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_img(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

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

        np.save(os.path.join(self.args.save_dir,'loss.npy'),np.array(self.loss_buffer))