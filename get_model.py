
from lib_unet import *

def get_model(class_name, inputs, targets=None, ngf=32,
              weight_loss=1, optimizer=tf.train.AdamOptimizer()):
    
    def caculate_loss(logits, targets, weight=20):
        targets = (targets+1)/2
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                        logits=logits, targets=targets, pos_weight=weight))
        return loss
    
    with tf.name_scope('scope_{}'.format(class_name)):
        with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
            logits = get_generator_unet(inputs, 1, ngf, 
                                        '{}_logits'.format(class_name), verbal=False)
            outputs = tf.sigmoid(logits)

        # Get all params (shared and exclusive)
        rt_dict = {'inputs' : inputs, 'outputs':outputs}
        if targets is not None:# Train mode
            rt_dict['targets'] = targets
            rt_dict['params'] = [var for var in tf.trainable_variables() 
                                         if 'unet' in var.name]
            for param in rt_dict['params']:
                if 'encode_output_' in param.name:
                    if not class_name in param.name:
                        rt_dict['params'].remove(param)
            # caculate loss        
            rt_dict['loss'] = caculate_loss(logits, targets, 1)

            if optimizer is not None:
                    rt_dict['grads'] = optimizer.compute_gradients(loss=rt_dict['loss'], 
                                           var_list=rt_dict['params'])
    return rt_dict
