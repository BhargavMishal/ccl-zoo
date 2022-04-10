from numpy.lib.function_base import _DIMENSION_NAME
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go


dataClass = {1:'Mammal', 2:'Bird', 3:'Reptile', 4:'Fish', 5:'Amphibian', 6:'Bug', 7:'Invertebrate'}
predictedClass={}
kmeans = KMeans(n_clusters=7, init='random')
pca = PCA(3)
metaBuffer = []
features = [ "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", 
            "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize"]



def loadData(path):
	df = pd.read_csv(path,header=None)
	df.columns = [ "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", 
            "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "label"]
	return df


def dimensionReduction(data):
	reducedDimData = pca.transform(data)
	return reducedDimData


def trainModel(data):
	labels = kmeans.fit_predict(data)
	return labels


def accuracy(data):
	global metaBuffer
	accuracy = 0
	for label in range(1,8):
		row = data[data['label'] == label]
		mostRepeatedLabel = max(list(row['kmeans_predicted']),key=list(row['kmeans_predicted']).count)
		hitPerClass = list(row['kmeans_predicted']).count(mostRepeatedLabel)
		perClassAccuracy = hitPerClass/len(row)
		accuracy += perClassAccuracy
		metaBuffer.append((perClassAccuracy, dataClass[label], label, mostRepeatedLabel))
	accuracy /= 7
	return accuracy


def mapPredictedLabelsToClasses(metaBuffer):
	global metaBufferSorted
	metaBufferSorted = sorted(metaBuffer, key = lambda x: x[0], reverse=True)
	mostRepeatedValues=[i[3] for i in metaBufferSorted]
	notMostRepeatedValues=[i for i in range(1,8) if i not in mostRepeatedValues]

	for i,d in enumerate(metaBufferSorted):
		if d[3] not in predictedClass:
			predictedClass[d[3]] = dataClass[d[2]]
	for i in notMostRepeatedValues:
		assigned = list(predictedClass.values())
		data = df[df['kmeans_predicted'] == i]
		a = max(list(data['label']),key = list(data['label']).count)
		if i not in predictedClass and dataClass[a] not in assigned:
			predictedClass[i] = dataClass[a]
			assigned.append(i)
		else:
			assign_t = set(assigned)
			all = set(['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate'])
			assign_f = list(all - assign_t)
			predictedClass[i]=random.choice(assign_f)


st.title("Unsupervised Zoo data classification")
st.subheader("An Unsupervised Approach using KMeans and PCA to classify animals based on their traits.")
st.write(" ")


st.markdown('''#### How To use this application ?

	1. 📦 To know more about the dataset and the contents of the dataset. Check below.
	2. ⚙️ Adjust the input fields in the left sidebar. _(click on > if closed)_ The results are displayed in the central
	3. 🔬 Inspect the scatter matrix and the central plots in the central widget.
	4. 📃You can view the accuracy of the whole model and also accuracy based on per target class basis.
	''')


df = loadData("./zoo_final.csv")

if st.checkbox('Explore the Dataset'):
	st.text('''This dataset consists of 101 animals from a zoo. There are 16 variables with various traits to describe the animals. \nThe 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate.''')
	st.text('''The purpose for this dataset is to be able to predict the classification of the animals, based upon their behaviour and traits.''')
	st.write(df)
	st.write(" ")


reducedDimData = pca.fit_transform(df.iloc[:,0:16])

predictedLabels = trainModel(reducedDimData)
df['kmeans_predicted'] = predictedLabels + 1
df['kmeans_class'] = predictedLabels + 1
for i in predictedClass:
	df['kmeans_class'].mask(df['kmeans_class'] == i, predictedClass[i], inplace=True)

acc = accuracy(df)

mapPredictedLabelsToClasses(metaBuffer)

pca_data = pd.DataFrame(reducedDimData)
pca_data['labels'] = predictedLabels + 1

for i in predictedClass:
	pca_data['labels'].mask(pca_data['labels'] == i, predictedClass[i], inplace=True)


if st.checkbox('View the Scatter Matrix'):
	fig = px.scatter_matrix(
		df,
		dimensions=features, color = 'label'
	)
	fig.update_traces(diagonal_visible=False)
	fig.update(layout_coloraxis_showscale=False) 
	fig.update_layout(title="Scatter Matrix", dragmode='select', width=1150, height=1150, hovermode='closest')
	st.plotly_chart(fig)
	st.write(" ")


totalVariance = pca.explained_variance_ratio_.sum() * 100
centroids = kmeans.cluster_centers_

fig = px.scatter_3d(pca_data, x=0, y=1, z=2, color = 'labels', symbol = 'labels', opacity = 0.7, size_max = 20, labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}, title=f'Total Explained Variance: {totalVariance:.2f}%')

fig1 = px.scatter_3d(centroids, x=0, y=1, z=2)

fig.update(layout_coloraxis_showscale=False) 
fig1.update(layout_coloraxis_showscale=False) 

fig3 = go.Figure(data=fig.data + fig1.data)
fig3.update(layout_coloraxis_showscale=False) 



with st.sidebar:
	st.header("Input Data")
	hair = st.selectbox("Does your animal have hair ?", ["Yes","No"],key="Hair")
	feathers = st.selectbox("Does it have feathers ?", ["Yes","No"],key="Feathers")
	eggs = st.selectbox("Can it lay eggs ?", ["Yes","No"],key="Eggs")
	milk = st.selectbox("Does it provide milk to the newborns ?", ["Yes","No"],key="Milk")
	airborne = st.selectbox("Can it fly ?", ["Yes","No"],key="Airborne")
	aquatic = st.selectbox("Can it swim ?", ["Yes","No"],key="Aquatic")
	predator = st.selectbox("Does it hunt ?", ["Yes","No"],key="Predator")
	toothed = st.selectbox("Does it contain teeth ?", ["Yes","No"],key="Toothed")
	backbone = st.selectbox("Does it have a backbone ?", ["Yes","No"],key="Backbone")
	breathes = st.selectbox("Does it breathe through it's nose ?", ["Yes","No"],key="Breathes")
	venomous = st.selectbox("Is it venomous ?", ["Yes","No"],key="Venomous")
	fins = st.selectbox("Does it have fins ?", ["Yes","No"],key="Fins")
	legs = st.number_input('How many legs does it has ?',value=0,min_value=0)
	tail = st.selectbox("Does it contain a tail ?", ["Yes","No"],key="Tail")
	domestic = st.selectbox("Is it a pet ?", ["Yes","No"],key="Domestic")
	catsize = st.selectbox("Is it the size of an average cat ?", ["Yes","No"],key="Catsize")


temp = [hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize]    

user_inputs = []
for i in temp:
	if i == "Yes":
		user_inputs.append(1)
	elif i == "No":
		user_inputs.append(0)
	else:
		user_inputs.append(i)
		
df_ip = pd.DataFrame([user_inputs])
pca_ip_data = pca.transform(df_ip.iloc[:,0:16])

if st.checkbox("Check out the accuracy "):
	st.write("Accuracy : ", acc)
	st.write("Per Class Accuracy")
	st.write([(i[1], i[0]) for i in metaBuffer])
	st.write(" ")

st.write(" ")
st.subheader("Predicted class:")
st.write(predictedClass[kmeans.predict(pca_ip_data)[0]+1])

pred_fig = px.scatter_3d(pca_ip_data, x=0, y=1, z=2,color = 2)
pred_fig.update(layout_coloraxis_showscale=False) 
pred_fig.update_traces(marker=dict(size=15, line=dict(width=2, color='Black')), selector=dict(mode='markers'))
fig3 = go.Figure(data=fig.data + fig1.data + pred_fig.data)
fig3.update(layout_coloraxis_showscale=False) 
st.plotly_chart(fig3)


st.markdown('''
#### Acknowledgements

UCI Machine Learning: https://archive.ics.uci.edu/ml/datasets/Zoo

Source Information
- Creator: Richard Forsyth
- Donor: Richard S. Forsyth
''')