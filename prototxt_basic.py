# prototxt_basic

import math

attrstr = "attrs"
#attrstr = "param"

names_output = {"rf_c2_upsampling":256 ,"rf_c3_upsampling":256}
#names_output = {"ssh_m2_red_up":32,"ssh_c3_up":32 }

def data(txt_file, info):
  txt_file.write('name: "mxnet-mdoel"\n')
  txt_file.write('layer {\n')
  txt_file.write('  name: "data"\n')
  txt_file.write('  type: "Input"\n')
  txt_file.write('  top: "data"\n')
  txt_file.write('  input_param {\n')
  txt_file.write('    shape: {{ dim: {} dim: {} dim: {} dim: {} }}\n'.format(info['shape'][0],
                                                                             info['shape'][1],
                                                                             info['shape'][2],
                                                                             info['shape'][3]))
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def fuzzy_haskey(d, key):
  for eachkey in d:
    if key in eachkey:
      return True
  return False
  
def Convolution(txt_file, info):
  print(info[attrstr])
  if fuzzy_haskey(info['params'], 'bias'):
    bias_term = 'true'  
  elif info[attrstr].has_key('no_bias') and info['attrs']['no_bias'] == 'True':

    bias_term = 'false'  
  else:
    bias_term = 'true'
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Convolution"\n')
  txt_file.write('  convolution_param {\n')
  txt_file.write('    num_output: %s\n'   % info[attrstr]['num_filter'])
  txt_file.write('    kernel_size: %s\n'  % info[attrstr]['kernel'].split('(')[1].split(',')[0]) # TODO
  if info[attrstr].has_key('pad'):
    txt_file.write('    pad: %s\n'          % info[attrstr]['pad'].split('(')[1].split(',')[0]) # TODO
  if info[attrstr].has_key('num_group'):
    txt_file.write('    group: %s\n'        % info[attrstr]['num_group'])
  if info[attrstr].has_key('stride'):
    txt_file.write('    stride: %s\n'       % info[attrstr]['stride'].split('(')[1].split(',')[0])
  if info[attrstr].has_key('dilate'):
    txt_file.write('    dilation: %s\n'       % info[attrstr]['dilate'].split('(')[1].split(',')[0])
  txt_file.write('    bias_term: %s\n'    % bias_term)
  txt_file.write('  }\n')
  if 'share' in info.keys() and info['share']:  
    txt_file.write('  param {\n')
    txt_file.write('    name: "%s"\n'     % info['params'][0])
    txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')  

def ChannelwiseConvolution(txt_file, info):
  Convolution(txt_file, info)
  
def BatchNorm(txt_file, info):
  #pprint.pprint(info)
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "BatchNorm"\n')
  txt_file.write('  batch_norm_param {\n')
  txt_file.write('    use_global_stats: true\n')        # TODO
  if info[attrstr].has_key('momentum'):
    txt_file.write('    moving_average_fraction: %s\n' % info[attrstr]['momentum'])
  else:
    txt_file.write('    moving_average_fraction: 0.9\n')
  if info[attrstr].has_key('eps'):
    txt_file.write('    eps: %s\n' % info[attrstr]['eps'])
  else:
    txt_file.write('    eps: 0.001\n')                   
  txt_file.write('  }\n')
  txt_file.write('}\n')
  # if info['fix_gamma'] is "False":                    # TODO
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['top'])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s_scale"\n'   % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param { bias_term: true }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Activation(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  if info[attrstr]['act_type']=='sigmoid':
    txt_file.write('  type: "Sigmoid"\n')
  else:
    txt_file.write('  type: "ReLU"\n')      # TODO
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Activation_Relu6(txt_file, info):
  info[attrstr]['act_type'] = 'ReLU'
  Activation(txt_file, info)
  pass

def Deconvolution(txt_file, info):
  if fuzzy_haskey(info['params'], 'bias'):
    bias_term = 'true'
  elif info[attrstr].has_key('no_bias') and info['attrs']['no_bias'] == 'True':
    bias_term = 'false'
  else:
    bias_term = 'true'
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Deconvolution"\n')
  txt_file.write('  convolution_param {\n')

  txt_file.write('    num_output: %s\n' % info[attrstr]['num_filter'])
  txt_file.write('    kernel_size: %s\n' % info[attrstr]['kernel'].split('(')[1].split(',')[0])  # TODO
  if info[attrstr].has_key('pad'):
    txt_file.write('    pad: %s\n' % info[attrstr]['pad'].split('(')[1].split(',')[0])  # TODO
  if info[attrstr].has_key('num_group'):
    txt_file.write('    group: %s\n' % info[attrstr]['num_group'])
  if info[attrstr].has_key('stride'):
    txt_file.write('    stride: %s\n' % info[attrstr]['stride'].split('(')[1].split(',')[0])
  if info[attrstr].has_key('dilate'):
    txt_file.write('    dilation: %s\n' % info[attrstr]['dilate'].split('(')[1].split(',')[0])
  txt_file.write('    bias_term: %s\n' % bias_term)

  txt_file.write('}\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Upsampling(txt_file, info):
  scale = int(info[attrstr]['scale'])
  assert(scale > 0)
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  #txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Deconvolution"\n')
  print(info[attrstr])
  print(info)
  txt_file.write('  convolution_param {\n')
  txt_file.write('    num_output: %s\n' % names_output[info["name"]])
  #txt_file.write('    num_output: %s\n' % info[attrstr]['num_filter'])
  txt_file.write('    kernel_size: %d\n' % (2 * scale - scale % 2))  # TODO
  txt_file.write('    stride: %d\n' % scale)
  txt_file.write('    pad: %d\n' % math.ceil((scale - 1)/2.0))  # TODO
  #txt_file.write('    group: %s\n' % info[attrstr]['num_filter'])
  txt_file.write('    group: %s\n' % names_output[info["name"]])

  txt_file.write('    bias_term: false\n')
  txt_file.write('    weight_filler: {\n')
  txt_file.write('      type: "bilinear"\n')
  txt_file.write('    }\n')

  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Concat(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Concat"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

#

def Crop(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  type: "Crop"\n')
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  for btom in info['bottom']:
      txt_file.write('  bottom: "%s"\n' % btom)
  txt_file.write('    crop_param { \n axis: 2    \n offset: 0 \n } \n' )
  txt_file.write('}\n')
  txt_file.write('\n')

def ElementWiseSum(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Eltwise"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  eltwise_param { operation: SUM }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Pooling(txt_file, info):
  pool_type = 'AVE' if info[attrstr]['pool_type'] == 'avg' else 'MAX'
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Pooling"\n')
  txt_file.write('  pooling_param {\n')
  txt_file.write('    pool: %s\n'         % pool_type)       # TODO  
  if info[attrstr].has_key('global_pool') and info[attrstr]['global_pool'] == 'True':
    txt_file.write('    global_pooling: true\n')
  else:
    txt_file.write('    kernel_size: %s\n'  % info[attrstr]['kernel'].split('(')[1].split(',')[0])
    txt_file.write('    stride: %s\n'       % info[attrstr]['stride'].split('(')[1].split(',')[0])
    if info[attrstr].has_key('pad'):
      txt_file.write('    pad: %s\n'          % info[attrstr]['pad'].split('(')[1].split(',')[0])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass


def FullyConnected(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "InnerProduct"\n')
  txt_file.write('  inner_product_param {\n')
  txt_file.write('    num_output: %s\n' % info[attrstr]['num_hidden'])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass



import json
def Reshape(txt_file, info):
  print(info)
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "Reshape"\n')
  g =eval(info["attrs"]["shape"])
  print("reshape",g)
  # exit()
  txt_file.write('  reshape_param { \nshape\n {dim: '+str(g[0])+'   \ndim: '+str(g[1])+'  \n dim:  '+str(g[2])+' \ndim: '+str(g[3])+' \n} \n}')


  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Flatten(txt_file, info):
  pass

def SoftmaxActivation(txt_file, info):
  # softmax
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'%(info['top']))
  txt_file.write('  name: "%s"\n'%(info['top']))
  txt_file.write('  type: "Softmax"\n')
  txt_file.write('  softmax_param: {\n')
  txt_file.write('    axis: 1\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')


def SoftmaxOutput(txt_file, info):
  # softmax
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "prob"\n')
  txt_file.write('  name: "prob"\n')
  txt_file.write('  type: "Softmax"\n')
  txt_file.write('  softmax_param: {\n')
  txt_file.write('    axis: 1\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

  # argmax
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "prob"\n')
  #txt_file.write('  top: "%s"\n' % info['top'])
  txt_file.write('  top: "out_label"\n')
  #txt_file.write('  name: "%s"\n' % info['top'])
  txt_file.write('  name: "out_label"\n')
  txt_file.write('  type: "ArgMax"\n')
  txt_file.write('  argmax_param: {\n')
  txt_file.write('    axis: 1\n')
  txt_file.write('    top_k: 1\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def LeakyReLU(txt_file, info):
  if info[attrstr]['act_type'] == 'elu':
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
    txt_file.write('  top: "%s"\n'        % info['top'])
    txt_file.write('  name: "%s"\n'       % info['top'])
    txt_file.write('  type: "ELU"\n')
    txt_file.write('  elu_param { alpha: 0.25 }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
  elif info[attrstr]['act_type'] == 'prelu':
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
    txt_file.write('  top: "%s"\n'        % info['top'])
    txt_file.write('  name: "%s"\n'       % info['top'])
    txt_file.write('  type: "PReLU"\n')
    txt_file.write('}\n')
    txt_file.write('\n')      
  else:
    raise Exception("unsupported Activation")

def Eltwise(txt_file, info, op):
  txt_file.write('layer {\n')
  txt_file.write('  type: "Eltwise"\n')
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  for btom in info['bottom']:
      txt_file.write('  bottom: "%s"\n' % btom)
  txt_file.write('  eltwise_param { operation: %s }\n' % op)
  txt_file.write('}\n')
  txt_file.write('\n')  

# ----------------------------------------------------------------
def write_node(txt_file, info):
    # info["top"] = info["top"].replace("_fwd","")
    if 'label' in info['name']:
        return        
    if info['op'] == 'null' and info['name'] == 'data':
        data(txt_file, info)
    elif info['op'] == 'Convolution':
        Convolution(txt_file, info)
    elif info['op'] == 'ChannelwiseConvolution':
        ChannelwiseConvolution(txt_file, info)
    elif info['op'] == 'BatchNorm':
        BatchNorm(txt_file, info)
    elif info['op'] == 'Activation':
        Activation(txt_file, info)
    elif info['op'] == 'ElementWiseSum':
        ElementWiseSum(txt_file, info)
    elif info['op'] == '_Plus':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'Concat':
        Concat(txt_file, info)
    elif info['op'] == 'Crop':
        Crop(txt_file,info)
    elif info['op'] == 'Pooling':
        Pooling(txt_file, info)
    elif info['op'] == 'Flatten':
        Flatten(txt_file, info)
    elif info['op'] == 'FullyConnected':
        FullyConnected(txt_file, info)
    elif info['op'] == 'SoftmaxOutput' or info['op'] == 'SoftmaxFocalOutput' :
        SoftmaxOutput(txt_file, info)
    elif info['op'] == 'LeakyReLU':
        LeakyReLU(txt_file, info)
    elif info['op'] == 'elemwise_add':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'UpSampling':
        Upsampling(txt_file, info)
    elif info['op'] == 'Deconvolution':
        Deconvolution(txt_file, info)
    elif info['op'] == 'clip':
        Activation_Relu6(txt_file, info)
    elif info['op'] == 'Reshape':
        Reshape(txt_file, info)
    elif info['op'] == 'SoftmaxActivation':
        SoftmaxActivation(txt_file, info)
        # pass
        # Activation_Relu6(txt_file, info)
    else:
        # pass
        print("unknown",info)
        # raise Exception("Warning! Skip Unknown mxnet op:{}".format(info['op']))





