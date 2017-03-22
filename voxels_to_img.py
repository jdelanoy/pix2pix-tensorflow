import tensorflow as tf
import numpy as np
import os

def interpolate(val, y0, x0, y1, x1 ):
    return (val-x0)*(y1-y0)/(x1-x0) + y0;


# def jet_value(gray):
#     with tf.name_scope("gray_to_jet"):
#         res = tf.identity(gray)
#         for i in range(res.get_shape()[0]):
#             for j in range(res.get_shape()[1]):
#                 val = res[i,j]
#                 if (val <= -0.75) is not None:
#                     tf.scatter_update(res, [[i,j]],[0])
#                     #res[i,j]= 0;
#                 elif (val <= -0.25) is not None:
#                     tf.scatter_update(res, [[i,j]],[0])
#                     #res[i,j]=interpolate( val, 0.0, -0.75, 1.0, -0.25 );
#                 elif (val <= 0.25) is not None:
#                     tf.scatter_update(res, [[i,j]],[0])
#                     #res[i,j]=1.0;
#                 elif (val <= 0.75) is not None:
#                     tf.scatter_update(res, [[i,j]],[0])
#                     #res[i,j]=interpolate( val, 1.0, 0.25, 0.0, 0.75 );
#                 else :
#                     tf.scatter_update(res, [[i,j]],[0])
#                     #res[i,j]= 0
#         return res


# def gray_to_jet(gray):
#     with tf.name_scope("jet_value"):
#         R = jet_value(tf.identity(gray)-0.5)
#         G = jet_value(tf.identity(gray))
#         B = jet_value(tf.identity(gray)+0.5)
#         return tf.stack([R,G,B],axis=2)


#input of sixe vox*vox*vox
def voxels_to_jet_depth(voxels, vox_size):
    with tf.name_scope("voxels_to_jet_depth"):
        mask = tf.cast(voxels>0.2, dtype=tf.float32)
        depth_I = tf.argmax(mask,2)
        depth = tf.to_float(depth_I)/vox_size;
        return tf.expand_dims(depth,-1)#gray_to_jet(depth)


#input of sixe vox*vox*vox
def voxels_to_grid(voxels, vox_size, nrows):
    with tf.name_scope("voxels_to_grid"):
        ncols = vox_size/nrows
        
        #size=64*8
        #gris = image[:,:,:,0]
        for i in range(nrows):
            img_tmp = voxels[:,:,i*8]
            #print(i)
            for j in range(1,int(ncols)):
                img_tmp =tf.concat([img_tmp,voxels[:,:,i*8+j]],0)
                #=tf.concat([vox,r[i:i+vox_size,j:j+vox_size,:]],2)
                #vox.append(r[i:i+vox_size,j:j+vox_size,:])
            #print (img_tmp.get_shape())
            if i ==0:
                res = img_tmp
            else:
                res = tf.concat([res,img_tmp],1)
            #print (res.get_shape())
        #imageI = tf.argmax(image,2)
        return tf.expand_dims(res,-1);

#input of sixe vox*vox*vox
def pack_voxels(voxels, vox_size, nrows):
    ncols = vox_size/nrows/4
    print ("entry pack ", voxels.get_shape())
        #img_tmp = voxels[:,:,i*8]
        #print(i)
    for i in range(nrows):
        for j in range(int(ncols)):
            for c in range(4):
            #img_tmp =tf.concat([img_tmp,voxels[:,:,i*(84+j]],0)
                if c == 0:
                    channel= tf.expand_dims(voxels[:,:,(j*nrows+i)*4+c],-1)
                else:
                    channel = tf.concat([channel,tf.expand_dims(voxels[:,:,(j*nrows+i)*4+c],-1)],2)
                #print ("c = ", c, channel.get_shape())
            if j ==0:
                row = channel
            else:
                row = tf.concat([row,channel],0)
            #print ("j = ", j, row.get_shape())
        if i ==0:
            img = row
        else:
            img = tf.concat([img,row],1)
        #print ("i = ", i, img.get_shape())
    print ("out pack ", img.get_shape())

    return img

#input of sixe vox*vox*vox
def unpack_voxels(img, vox_size, nrows):
    ncols = vox_size/nrows/4
    print ("entry unpack ", img.get_shape())

    vox=img[0:vox_size,0:vox_size,:]
    for i in range(0,nrows*vox_size,vox_size):
        for j in range(0,int(ncols)*vox_size,vox_size):
            vox=tf.concat([vox,img[i:i+vox_size,j:j+vox_size,:]],2)
            #print(i," ",j," => ",vox.get_shape())
    vox = vox[:,:,4:]
    print ("out unpack ", vox.get_shape())

    return vox
