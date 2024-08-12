from google.colab import drive
import os
drive.mount('/content/drive')
drive_folder='drive/MyDrive/TESIS/'
tags_file=drive_folder+'db/data.csv'
data_folder=drive_folder+'db/data/'
image_folder=drive_folder+'db/images/cow/'
shape_folder=drive_folder+'db/images/shape/'
output_folder=drive_folder+'db/output/'
test_output_folder=drive_folder+'db/output-test/'
files=os.listdir(image_folder)
import os,torch,torchvision.transforms as T,torchvision.models as models,matplotlib.pyplot as plt
from PIL import Image
import numpy as np,torchvision.models.segmentation as segmentation_models
available_models=dir(segmentation_models)
available_models=['deeplabv3','deeplabv3_mobilenet_v3_large','deeplabv3_resnet101','deeplabv3_resnet50','fcn','fcn_resnet101','fcn_resnet50','lraspp','lraspp_mobilenet_v3_large']
model_name='lraspp_mobilenet_v3_large'
image_folder=image_folder
pretrained=True
resize_input=256
crop_size=224
mean=[.485,.456,.406]
std=[.229,.224,.225]
transform=T.Compose([T.Resize(resize_input),T.CenterCrop(crop_size),T.ToTensor(),T.Normalize(mean=mean,std=std)])
def getTransform(image_folder):
	import torchvision.transforms as T,numpy as np;from PIL import Image;import os;from collections import Counter;pixel_values=[];dimensions=[]
	for image_name in os.listdir(image_folder):
		if image_name.lower().endswith(('.jpg','.jpeg','.png')):image_path=os.path.join(image_folder,image_name);image=Image.open(image_path).convert('RGB');dimensions.append(image.size);image=np.array(image);pixel_values.append(image)
	most_common_dimensions=Counter(dimensions).most_common(1)[0][0];resize_input=min(most_common_dimensions);crop_size=min(most_common_dimensions);pixel_values=np.array(pixel_values);mean=np.mean(pixel_values,axis=(0,1,2))/255.;std=np.std(pixel_values,axis=(0,1,2))/255.;transform=T.Compose([T.Resize(resize_input),T.CenterCrop(crop_size),T.ToTensor(),T.Normalize(mean=mean.tolist(),std=std.tolist())]);return transform
transform=getTransform(image_folder)
model_fn=getattr(models.segmentation,model_name)
model=model_fn(pretrained=pretrained)
model.eval()
def save_image(extracted_image,image_name,test=False):
	output=test_output_folder if test else output_folder
	if not os.path.exists(output):os.makedirs(output)
	output_path=os.path.join(output,image_name);extracted_image.save(output_path)
def show_image(original_image,extracted_image,accuracy_percentage=''):fig,ax=plt.subplots(1,2,figsize=(12,6));ax[0].imshow(original_image);ax[0].set_title('Original Image');ax[0].axis('off');ax[1].imshow(extracted_image);ax[1].set_title(f"Extracted Region ({round(accuracy_percentage,2)}%)");ax[1].axis('off');plt.show()
def export_predictions_to_csv(results,filename='predictions.csv',output_file=''):
	import csv
	if not os.path.exists(output_folder+output_file):os.makedirs(output_folder+output_file)
	output=output_folder+output_file+'/'+filename
	with open(output,mode='w',newline='')as file:
		writer=csv.writer(file);writer.writerow(['Image Name','Prediction Accuracy (%)','Mean Distance'])
		for result in results:writer.writerow(result)
def segment_image(image,model,transform):
	input_image=transform(image).unsqueeze(0)
	with torch.no_grad():output=model(input_image)['out']
	output_predictions=output.argmax(1)[0];return output_predictions
def extract_region(image,mask):original_size=image.size;mask_resized=mask.byte().cpu().numpy();mask_resized=Image.fromarray(mask_resized).resize(original_size,resample=Image.NEAREST);mask_resized=np.array(mask_resized);image_np=np.array(image);extracted_image=image_np*np.expand_dims(mask_resized,axis=2);return Image.fromarray(extracted_image.astype(np.uint8))
def preprocess_image(image_name,model,transform):
	if image_name.endswith('.jpg')or image_name.endswith('.png'):image_path=os.path.join(image_folder,image_name);original_image=Image.open(image_path).convert('RGB');mask=segment_image(original_image,model,transform);extracted_image=extract_region(original_image,mask);return original_image,extracted_image
import pandas as pd
def read_points(point_name):
	point_path=os.path.join(data_folder,point_name)
	with open(point_path,'r')as f:points=f.readlines();return points
data_df=pd.read_csv(tags_file)
data_df=data_df.drop_duplicates()
data_df.head()
import torch.nn.functional as F
from scipy import ndimage
def get_centroids(mask):mask_np=np.array(mask);labels,num_labels=ndimage.label(mask_np);centroids=ndimage.center_of_mass(mask_np,labels,np.arange(1,num_labels+1));return centroids
def calculate_accuracy_percentage(mean_distance,image_size):max_distance=max(image_size);accuracy=max(0,1-mean_distance/max_distance)*100;return accuracy
def evaluate_prediction(image_name,point_name,model,transform,image_folder,data_folder):
	image_path=os.path.join(image_folder,image_name);image=Image.open(image_path).convert('RGB');mask=segment_image(image,model,transform);point_path=os.path.join(data_folder,point_name)
	with open(point_path,'r')as f:points=[list(map(float,line.split()))for line in f]
	predicted_points=get_centroids(mask);distances=[]
	for real_point in points:
		real_point=np.array(real_point);min_distance=float('inf')
		for pred_point in predicted_points:
			pred_point=np.array(pred_point);distance=np.linalg.norm(real_point-pred_point)
			if distance<min_distance:min_distance=distance
		distances.append(min_distance)
	mean_distance=np.mean(distances);accuracy_percentage=calculate_accuracy_percentage(mean_distance,image.size);return mean_distance,accuracy_percentage
def evaluate_all_images(data_df,image_folder,data_folder,model,transform,export_output='',show_image=False):
	results=[]
	for(idx,row)in data_df.iterrows():
		image_name=row['IMAGE'];shape_name=row['SHAPE'];point_name=row['POINT'];mean_distance,accuracy_percentage=evaluate_prediction(image_name,point_name,model,transform,image_folder,data_folder);results.append((image_name,accuracy_percentage,mean_distance))
		if show_image or export_output!='':original_image,extracted_image=preprocess_image(image_name,model,transform)
		if show_image:show_image(original_image,extracted_image,accuracy_percentage)
		if export_output!='':save_image(extracted_image,f"{export_output}/{image_name}")
	if export_output!='':export_predictions_to_csv(results,filename='predictions.csv',output_file=export_output)
	return results
def evaluate_random_image(data_df,image_folder,data_folder,model,transform,num_samples=1,export_output='',show_image=False):
	if int(num_samples)<1 or int(num_samples)>len(data_df):raise ValueError('num_samples debe estar entre 1 y el número total de filas en data_df')
	import random;random_rows=data_df.sample(n=num_samples);results=evaluate_all_images(random_rows,image_folder,data_folder,model,transform,export_output,show_image);return results
num_samples=1
export_output=''
show_image=False
results=evaluate_random_image(data_df,image_folder,data_folder,model,transform,num_samples,export_output,show_image)
results=evaluate_all_images(data_df,image_folder,data_folder,model,transform,export_output,show_image)
def plot_accuracy_scatter(results,show=True):
	results.sort(key=lambda x:x[0]);image_names=[result[0]for result in results];accuracies=[result[1]for result in results];plt.figure(figsize=(12,6));plt.scatter(image_names,accuracies,color='blue',alpha=.6,edgecolors='w',s=100);plt.xticks([]);plt.ylabel('Accuracy Percentage');plt.title('Accuracy Percentage of Model Predictions for Each Image');plt.ylim(0,100);plt.grid(axis='y',linestyle='--',alpha=.7);plt.tight_layout()
	if show:plt.show()
	return plt
if'results'in locals():plot_accuracy_scatter(results)
def plot_accuracy_pie(results,show=True):
	results.sort(key=lambda x:x[0]);accuracies=[result[1]for result in results];good_average_accuracy=sum(accuracies)/len(accuracies);bad_average_accuracy=100-good_average_accuracy;labels=['Predicted  Accuracy ({}%)'.format(good_average_accuracy),'Predicted Error ({}%)'.format(bad_average_accuracy)];sizes=[good_average_accuracy,bad_average_accuracy];colors=['lightgreen','lightcoral'];plt.figure(figsize=(8,8));plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=140);plt.axis('equal');plt.title('Overall Accuracy Distribution')
	if show:plt.show()
	return plt
if'results'in locals():plot_accuracy_pie(results)
def get_model(model_name):
	model_name=model_name.lower().split('_weights')[0];model_mapping={'deeplabv3':models.segmentation.deeplabv3_resnet50,'deeplabv3_resnet50':models.segmentation.deeplabv3_resnet50,'deeplabv3_resnet101':models.segmentation.deeplabv3_resnet101,'deeplabv3_mobilenet_v3_large':models.segmentation.deeplabv3_mobilenet_v3_large,'fcn':models.segmentation.fcn_resnet50,'fcn_resnet50':models.segmentation.fcn_resnet50,'fcn_resnet101':models.segmentation.fcn_resnet101,'lraspp':models.segmentation.lraspp_mobilenet_v3_large,'lraspp_mobilenet_v3_large':models.segmentation.lraspp_mobilenet_v3_large};weights_mapping={'deeplabv3':models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,'deeplabv3_resnet50':models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,'deeplabv3_resnet101':models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,'deeplabv3_mobilenet_v3_large':models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,'fcn':models.segmentation.FCN_ResNet50_Weights.DEFAULT,'fcn_resnet50':models.segmentation.FCN_ResNet50_Weights.DEFAULT,'fcn_resnet101':models.segmentation.FCN_ResNet101_Weights.DEFAULT,'lraspp':models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT,'lraspp_mobilenet_v3_large':models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT}
	if model_name not in model_mapping:print(f"Depuración: Modelos disponibles - {list(model_mapping.keys())}");raise ValueError(f"Model {model_name} is not supported.")
	model_fn=model_mapping[model_name];weights=weights_mapping.get(model_name,None);return model_fn(weights=weights)
def compare_models(available_models,data_df,csv_filename='compare_models'):
	import time,csv;csv_path=f"{output_folder}/{csv_filename}.csv";data_df=data_df.drop_duplicates()
	if os.path.exists(csv_path):existing_results=pd.read_csv(csv_path)
	else:
		with open(csv_path,mode='w',newline='')as file:writer=csv.writer(file);dfcolumns=['model_name','image_name','accuracy_percentage','mean_distance','execution_time'];writer.writerow(dfcolumns)
		existing_results=pd.DataFrame(columns=dfcolumns)
	model_results={}
	for model_name in available_models:
		try:
			model=get_model(model_name);model.eval();print(f">>> Evaluating model: {model_name} in {len(data_df)} images");total_execution_time=0;local_result=[];model_output_folder=os.path.join(output_folder,model_name)
			if not os.path.exists(model_output_folder):os.makedirs(model_output_folder)
			if not existing_results.empty:evaluated_images=existing_results[existing_results['model_name']==model_name]['image_name'].tolist();data_df_filtered=data_df[~data_df['IMAGE'].isin(evaluated_images)];print(f"Skipping {len(data_df)-len(data_df_filtered)} images already evaluated.")
			else:data_df_filtered=data_df
			previous_results=existing_results[existing_results['model_name']==model_name]
			for(_,row)in previous_results.iterrows():
				image_name=row['image_name'];accuracy_percentage=row['accuracy_percentage'];mean_distance=row['mean_distance'];execution_time=row['execution_time']
				if model_name not in model_results:model_results[model_name]=[]
				model_results[model_name].append((image_name,accuracy_percentage,mean_distance,execution_time))
			if data_df_filtered.empty:print(f"Skipping {model_name}, all images already evaluated.");continue
			i_counter=0
			for(idx,row)in data_df_filtered.iterrows():
				image_name=row['IMAGE'];point_value=row['POINT'];i_counter+=1
				try:
					start_time=time.time();mean_distance,accuracy_percentage=evaluate_prediction(image_name,point_value,model,transform,image_folder,data_folder);local_result.append((image_name,accuracy_percentage,mean_distance));original_image,extracted_image=preprocess_image(image_name,model,transform);save_image(extracted_image,f"{model_name}/{image_name}");end_time=time.time();execution_time=end_time-start_time;total_execution_time+=execution_time
					with open(csv_path,mode='a',newline='')as file:writer=csv.writer(file);writer.writerow([model_name,image_name,accuracy_percentage,mean_distance,execution_time])
					if model_name not in model_results:model_results[model_name]=[]
					model_results[model_name]=[image_name,accuracy_percentage,mean_distance,execution_time];print(f"[{i_counter}/{len(data_df_filtered)}] :: Model {model_name} - Image ID: {idx+2} ({image_name}) evaluation completed in {execution_time:.2f} seconds.")
				except Exception as e:print(f"Error evaluating image {image_name} with model {model_name}: {e}")
			if local_result:plt1=plot_accuracy_pie(local_result,show=False);plt1.savefig(f"{os.path.join(output_folder,model_name)}_pie.png");plt2=plot_accuracy_scatter(local_result,show=False);plt2.savefig(f"{os.path.join(output_folder,model_name)}_scatter.png")
			print(f"🚩 Model {model_name} evaluation completed in {total_execution_time:.2f} seconds.")
		except AttributeError:print(f"Model {model_name} not found in torchvision.models.segmentation.")
	print('🎉 Finished ....');return model_results
import pandas as pd
data_df=pd.read_csv(tags_file)
transform=getTransform(image_folder)
if'model'in locals():del model
model_results=compare_models(available_models,data_df)
def plot_model_comparison(model_results):
	import pandas as pd,matplotlib.pyplot as plt;from statistics import mean;df_results=pd.DataFrame({'Model':[model for model in model_results],'Accuracy (%)':[mean([result[1]for result in model_results[model]])for model in available_models],'Execution Time (s)':[sum([result[3]for result in model_results[model]])for model in available_models]});plt.figure(figsize=(14,6));plt.subplot(1,2,1);bars=plt.bar(df_results['Model'],df_results['Accuracy (%)'],color='skyblue');plt.xlabel('Model');plt.ylabel('Accuracy (%)');plt.title('Model Accuracy Comparison');plt.xticks(rotation=45,ha='right');plt.yticks(range(0,101,10))
	for bar in bars:yval=bar.get_height();plt.annotate(f"{yval}%",xy=(bar.get_x()+bar.get_width()/2,yval),xytext=(0,-125),textcoords='offset points',ha='center',va='bottom',rotation=90)
	plt.subplot(1,2,2);bars=plt.bar(df_results['Model'],df_results['Execution Time (s)'],color='salmon');plt.xlabel('Model');plt.ylabel('Execution Time (s)');plt.title('Model Execution Time Comparison');plt.xticks(rotation=45,ha='right');min_time=min(df_results['Execution Time (s)']);max_time=max(df_results['Execution Time (s)']);ymin=min_time-8;ymax=max_time+1001;yminlim=min_time-100;ymaxlim=max_time+1000;ystep=450;plt.ylim(yminlim,max_time+500);plt.yticks(range(int(ymin),int(ymax),ystep))
	for bar in bars:yval=bar.get_height();plt.annotate(f"{yval:.2f}",xy=(bar.get_x()+bar.get_width()/2,yval),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',rotation=0)
	plt.tight_layout();plt.show()
plot_model_comparison(model_results)
def plot_model_performance(model_results):
	import pandas as pd,matplotlib.pyplot as plt;from statistics import mean;df_results=pd.DataFrame({'Model':[model for model in model_results],'Accuracy (%)':[mean([result[1]for result in model_results[model]])for model in available_models],'Execution Time (s)':[sum([result[3]for result in model_results[model]])for model in available_models]});df_results['Performance Index']=df_results['Accuracy (%)']/df_results['Execution Time (s)']*100000;df_results_sorted=df_results;plt.figure(figsize=(10,6));bars=plt.bar(df_results_sorted['Model'],df_results_sorted['Performance Index'],color='lightgreen');plt.xlabel('Model');plt.ylabel('Performance Index');plt.title('Model Performance Comparison');plt.xticks(rotation=45,ha='right');min_performance=df_results_sorted['Performance Index'].min();max_performance=df_results_sorted['Performance Index'].max();ymax=max_performance+6299+1;ymin=min_performance-780;yminlim=min_performance-100;ymaxlim=max_performance+1000;ystep=5000;plt.ylim(yminlim,ymaxlim);plt.yticks(range(int(ymin),int(ymax),ystep))
	for bar in bars:yval=bar.get_height();plt.annotate(f"{yval:.0f}",xy=(bar.get_x()+bar.get_width()/2,yval),xytext=(0,2),textcoords='offset points',ha='center',va='bottom',rotation=0)
	plt.tight_layout();plt.show()
plot_model_performance(model_results)