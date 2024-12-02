import torch.onnx 
import onnxruntime as ort
import numpy as np
from model.model_culane import parsingNet
from onnx import checker

dummy_input = torch.randn(1, 3, 320, 1600)
test_input = torch.randn(1, 3, 320, 1600)  ##get the different input from converting process , verification test
path = './culane_res18.pth'
out_path = './culane_res18.onnx'
model = parsingNet(backbone='18',num_grid_row=200,num_cls_row=72,num_lane_on_row=4,num_grid_col=100,num_cls_col=81,num_lane_on_col=4,input_width=1600,input_height=320)  

state_dict = torch.load(path, map_location='cpu')['model']

##remove no need string 'model' form name of each layer
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

model.load_state_dict(compatible_state_dict, strict=False)
model.eval()

# PyTorch reference output
with torch.no_grad():
    out = model(test_input)

model.eval()
# export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    out_path,
    export_params=True,  # store the trained parameter weights inside the model file 
    opset_version=11,    # the ONNX version to export the model to 
    input_names = ['input'],   # the model's input names 
    output_names = ['loc_row','loc_col','exist_row','exist_col'], # the model's output names 
)

# ONNX reference output
ort_session = ort.InferenceSession(out_path,providers = ['CUDAExecutionProvider'])
outputs = ort_session.run(
    None,
    {"input": test_input.numpy()},
)
checker.check_model(out_path, True)

# compare ONNX Runtime and PyTorch results
t1,t2,t3,t4=out['loc_row'],out['loc_col'],out['exist_row'],out['exist_col']
print(np.max(np.abs(np.array(t1.detach().numpy()) - outputs[0])))
print(np.max(np.abs(np.array(t2.detach().numpy()) - outputs[1])))
print(np.max(np.abs(np.array(t3.detach().numpy()) - outputs[2])))
print(np.max(np.abs(np.array(t4.detach().numpy()) - outputs[3])))