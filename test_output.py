import tensorflow as tf
import numpy as np
import os

from subprocess import call

def save_images(fetches, output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets", "outputs_grid", "targets_grid","outputs_packed","targets_packed"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        for kind in ["outputs","targets"]:
            in_kind = kind + "_packed"
            out_kind = kind + "_normals"
            filename = name + "-" + out_kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[out_kind] = filename
            out_path = os.path.join(image_dir, filename)
            in_path = os.path.join(image_dir, name + "-" + in_kind + ".png")
            call(['/user/delanoy/home/code/voxels/src/build/write_normal_map',in_path,out_path,'128'])
        filesets.append(fileset)
    return filesets

def init_table(index, step=False):
    index.write("<table><tr>")
    if step:
        index.write("<th>step</th>")
    index.write("<th>name</th>")
    for kind in ["inputs", "outputs", "targets", "outputs_grid", "targets_grid", "outputs_normals", "targets_normals"]:
        index.write("<th>%s</th>" % kind)
    index.write("</tr>\n")
    return

    
def append_index(filesets, index, step=False):
    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets", "outputs_normals", "targets_normals", "outputs_grid", "targets_grid"]:
            index.write("<td><img src='images/%s' width='256' height='256'></td>" % fileset[kind])

        index.write("</tr>\n")
    return 


