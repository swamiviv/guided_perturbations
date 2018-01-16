import os,sys
import caffe
caffe.set_mode_gpu()
from PIL import Image
from datetime import datetime
import numpy as np
import timeit
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import copy
import surgery
import argparse


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in xrange(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def colorize(label,palette):
    out = Image.fromarray(label)
    out.putpalette(palette)
    return out


def preprocess(im):
    mean = (104.00698793, 116.66876762, 122.67891434) 
    in_ = np.array(copy.deepcopy(im), dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= mean
    in_ = in_.transpose((2,0,1))
    return in_[np.newaxis,...]


def main():

    palette = get_palette(256)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_file',default='input.png', help='Input image')
    parser.add_argument('-e','--eps', type=float,default=0.55, help='Epsilon value')
    parser.add_argument('-w','--model_weights',required=True, help='Model weights')
    parser.add_argument('-p','--model_proto',required=True,help='Model prototxt file')
    parser.add_argument('-t','--type',default='same',help='Type of output modifier (one-hot or same) ')
    parser.add_argument('-o','--outfile', default='output.png')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    caffe.set_device(args.gpu)
    
    net=caffe.Net(args.model_proto,args.model_weights,caffe.TEST)
    # surgeries
    interp_layers = [k for k in net.params.keys() if 'up' in k]
    surgery.interp(net, interp_layers)

    mean = (104.00698793, 116.66876762, 122.67891434) 
    epsilon = args.eps

    img = Image.open(args.input_file)
    in_batch = preprocess(img)
       
           

    net.blobs['data'].reshape(*in_batch.shape)
    net.blobs['data'].data[...] = in_batch
    net.forward()                
    out = copy.deepcopy(net.blobs[net.outputs[0]].data[0])
    out_argmax = out.argmax(0)


            
    #Modify output to backprop gradient based on network output
    sh = out.squeeze().shape
    conf_softmax = np.zeros((1,sh[0],sh[1],sh[2]))        
    if args.type == 'one-hot':
        for i in np.arange(sh[0]):
            for j in np.arange(sh[1]):                                                    
                conf_softmax[0,out_argmax[i,j],i,j]=1.0  
    elif args.type== 'same':
        conf_softmax = copy.deepcopy(out[None,:,:,:])



    sh = net.blobs['data'].diff[...].shape
    net.blobs['data'].diff[...] = np.zeros(sh)


    
    #Perform backward pass - no updates happen here
    net.backward(**{net.outputs[0]: conf_softmax})
    

    #Read of the diff from data layer - this is the perturbation
    perturbed_data = (float(args.eps)*np.sign(net.blobs['data'].diff[...])+ in_batch)


    #Run the forward pass using the perturbed_input
    net.blobs['data'].reshape(*perturbed_data.shape)
    net.blobs['data'].data[...] = perturbed_data
    net.forward()

    #Obtain the final output
    out_gp = net.blobs[net.outputs[0]].data[0]
    out_gp_argmax = out_gp.argmax(0)


    #Colorize the label outputs and write to file
    out_color = colorize(out_argmax.astype(np.uint8),palette)
    out_color_gp = colorize(out_gp_argmax.astype(np.uint8),palette)
    w,h = out_color.size
    out_comb = Image.new('RGB',(3*w,h))

    out_comb.paste(img,(0,0))
    out_comb.paste(out_color,(w,0))
    out_comb.paste(out_color_gp,(2*w,0))
    out_comb.save(args.outfile)


if __name__ == '__main__':
    main()

