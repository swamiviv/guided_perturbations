model_name='fcn32s.caffemodel'
model_url='http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel'
model_proto='deploy32s.prototxt'
#Download weights
#wget -O $model_name $model_url
python demo.py -w $model_name -p $model_proto -g 3

