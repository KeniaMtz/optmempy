##
#  @file itils.py
#
#  @brief Utils for training of artificial neural network in Keras and export to file that is readable by MeLOn.
#
# ==============================================================================\n
#   Aachener Verfahrenstechnik-Systemverfahrenstechnik, RWTH Aachen University  \n
# ==============================================================================\n
#
#  @author Artur M. Schweidtmann, Friedrich von Bülow, Jing Cui, Laurens Lueg, and Alexander Mitsos
#  @date 20. January 2020
#  https://github.com/ddceruti/maingo_example/blob/main/utils.py
##

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xml.etree.cElementTree as ElemTree
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import json
import torch
import maingopy

def save_model_to_xml(filename, stripped_model, X, y, scaleInput, normalizeOutput, precision_to_xml=30):
        n_neurons = []
        
        input_lower_bound = np.amin(X, axis=0)
        input_upper_bound = np.amax(X, axis=0)
        output_lower_bound = np.amin(y, axis=0)
        output_upper_bound = np.amax(y, axis=0)
        y_mean = np.mean(y, axis=0) 
        y_std = np.std(y,axis=0)

        ffnet = ElemTree.Element("FFNET")
        description = ElemTree.SubElement(ffnet, "Description")
        values = ElemTree.SubElement(ffnet, "Values")
        inputIsScaled = ElemTree.SubElement(values, "inputIsScaled")
        outputIsNormalized = ElemTree.SubElement(values, "outputIsNormalized")
        std_of_output = ElemTree.SubElement(values, "Stdofoutput")
        mean_of_output = ElemTree.SubElement(values, "Meanofoutput")
        architecture = ElemTree.SubElement(values, "Architecture")
        bounds = ElemTree.SubElement(values, "Bounds")
        layers = ElemTree.SubElement(values, "Layers")
        
        precision_to_xml_str = "{0:." + str(precision_to_xml) + "f}"

        # Description
        ElemTree.SubElement(description, "Description", name="Description").text = str("Optional Description")

        #Information about data preprocessing
        ElemTree.SubElement(inputIsScaled, "inputIsScaled", name="inputIsScaled").text = str(int(scaleInput))
        ElemTree.SubElement(outputIsNormalized, "outputIsNormalized", name="outputIsNormalized").text = str(int(normalizeOutput))
        
        # Standard deviation and mean of outputs
        #ElemTree.SubElement(std_of_output, "outputStd", name="outputStd").text = str(float(y_std))
        #ElemTree.SubElement(mean_of_output, "outputMean", name="outputMean").text = str(float(y_mean))
        # Standard deviation of outputs
        if isinstance(y_std, np.ndarray):
            for i, val in enumerate(y_std):
                ElemTree.SubElement(
                    std_of_output,
                    "outputStd",
                    name=f"outputStd_{i}"
                ).text = str(val)
        else:
            ElemTree.SubElement(std_of_output, "outputStd", name="outputStd").text = str(y_std)

        # Mean of outputs
        if isinstance(y_mean, np.ndarray):
            for i, val in enumerate(y_mean):
                ElemTree.SubElement(
                    mean_of_output,
                    "outputMean",
                    name=f"outputMean_{i}"
                ).text = str(val)
        else:
            ElemTree.SubElement(mean_of_output, "outputMean", name="outputMean").text = str(y_mean)


        # Architecture Iterate over layers to get count number of dense layers (Dropout layers are not being counted)
        number_of_layers = 0
        for layerIndex, modelLayer in enumerate(stripped_model.layers):
            if isinstance(modelLayer, tf.keras.layers.Dense):
                number_of_layers += 1

        ElemTree.SubElement(architecture, "NumberOfLayers", name="NumberOfLayers").text = str(number_of_layers)
        ElemTree.SubElement(architecture, "NumberOfInputs", name="NumberOfInputs").text = \
            str(len(stripped_model.layers[0].get_weights()[0]))
        if isinstance(stripped_model.layers[len(stripped_model.layers) - 1],  tf.keras.layers.Dense):
            # if last layer is dense, use it to get the NumberOfOutputs
            number_of_outputs = len(stripped_model.layers[len(stripped_model.layers) - 1].get_weights()[1])
        else:
            # else: use the 2ndlast layer is dense, use it to get the NumberOfOutputs
            number_of_outputs = len(stripped_model.layers[len(stripped_model.layers) - 2].get_weights()[1])
        ElemTree.SubElement(architecture, "NumberOfOutputs", name="NumberOfOutputs").text = str(number_of_outputs)

        # Iterate over Bounds (Arrays)
        input_lb = ElemTree.SubElement(bounds, "InputLowerBound")
        if isinstance(input_lower_bound, np.ndarray):
            for boundIndex, bound in enumerate(input_lower_bound):
                ElemTree.SubElement(input_lb, "Bounds", name="inputLB_" + str(boundIndex)).text = str(bound)
        else:
            ElemTree.SubElement(input_lb, "Bounds", name="inputLB_").text = str(input_lower_bound.values[0])
        input_ub = ElemTree.SubElement(bounds, "InputUpperBound")
        if isinstance(input_upper_bound, np.ndarray):
            for boundIndex, bound in enumerate(input_upper_bound):
                ElemTree.SubElement(input_ub, "Bounds", name="inputUB_" + str(boundIndex)).text = str(bound)
        else:
            ElemTree.SubElement(input_ub, "Bounds", name="inputUB_").text = str(input_upper_bound.values[0])
        output_lb = ElemTree.SubElement(bounds, "OutputLowerBound")
        if isinstance(output_lower_bound, np.ndarray):
            for boundIndex, bound in enumerate(output_lower_bound):
                ElemTree.SubElement(output_lb, "Bounds", name="outputLB_" + str(boundIndex)).text = str(bound)
        else:
            ElemTree.SubElement(output_lb, "Bounds", name="outputLB").text = str(output_lower_bound.values[0])
        output_ub = ElemTree.SubElement(bounds, "OutputUpperBound")
        if isinstance(output_upper_bound, np.ndarray):
            for boundIndex, bound in enumerate(output_upper_bound):
                ElemTree.SubElement(output_ub, "Bounds", name="outputUB_" + str(boundIndex)).text = str(bound)
        else:
            ElemTree.SubElement(output_ub, "Bounds", name="outputUB_").text = str(output_upper_bound.values[0])

        dense_layer_counter = 0
        # Iterate over layer to get weights and biases
        for layerIndex, modelLayer in enumerate(stripped_model.layers):
            if isinstance(modelLayer,  tf.keras.layers.Dense):
                layer = ElemTree.SubElement(layers, "Layer_" + str(dense_layer_counter))
                # Write Number of Neurons per Layer while iterating over all layers. Number of neurons per layer
                # equals number of biases per layer.
                if dense_layer_counter + 1 < number_of_layers:
                    n_neurons.append(len(modelLayer.get_weights()[1]))
                # Write Activation Function
                ElemTree.SubElement(layer, "ActivationFunction",
                                    name="ActivationFunction_" + str(dense_layer_counter)).text = str(
                    modelLayer.activation.__name__)
                # Write Biases
                bias = ElemTree.SubElement(layer, "Bias")
                for index2, singleBias in enumerate(modelLayer.get_weights()[1]):
                    ElemTree.SubElement(bias, "Bias",
                                        name="Bias_" + str(dense_layer_counter) + "_" + str(
                                            index2)).text = precision_to_xml_str.format(singleBias)
                # Write weights
                weights = ElemTree.SubElement(layer, "Weights")
                for index3, vectorWeights in enumerate(modelLayer.get_weights()[0]):
                    for index4, singleWeight in enumerate(vectorWeights):
                        ElemTree.SubElement(weights, "Weight",
                                            name="Weight_" + str(dense_layer_counter) + "_" + str(index3) + "_" + str(
                                                index4)).text = precision_to_xml_str.format(singleWeight)
                dense_layer_counter += 1

        layer_sizes = ElemTree.SubElement(architecture, "LayerSizes")
        for neuron in n_neurons:
            ElemTree.SubElement(layer_sizes, "LayerSize", name="LayerSize").text = str(neuron)

        tree = ElemTree.ElementTree(ffnet)

        tree.write(filename)

def scale(X, scaleInput):
    # scale Input values to range [-1,1] in each dimension
    if (scaleInput):
        nom = (X -  X.min(axis=0))*2
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return -1 + nom/denom
    else:
        return X

def normalize(y, normalizeOutput):
    # normalize output to z-score
    if(normalizeOutput):
        y_norm = (y - np.mean(y, axis=0))/np.std(y, axis=0);
        return y_norm
    else:
        return y


def generate_melon_scaler_object(scaler):
        scaler_data = maingopy.melonpy.ScalerData()
        
        if scaler is None:
            scaler_data = maingopy.melonpy.SCALER_TYPE.IDENTITY
            scaler_data.parameters = {}
        elif isinstance(scaler, MinMaxScaler):
            scaler_data.type = maingopy.melonpy.SCALER_TYPE.MINMAX
            
            scaled_bounds = scaler.get_params()['feature_range']
            scaler_data.parameters = {
                maingopy.melonpy.SCALER_PARAMETER.UPPER_BOUNDS : scaler.data_max_.tolist(),
                maingopy.melonpy.SCALER_PARAMETER.LOWER_BOUNDS : scaler.data_min_.tolist(),
                maingopy.melonpy.SCALER_PARAMETER.SCALED_LOWER_BOUNDS : [scaled_bounds[0]]*len(scaler.data_max_.tolist()), 
                maingopy.melonpy.SCALER_PARAMETER.SCALED_UPPER_BOUNDS : [scaled_bounds[1]]*len(scaler.data_max_.tolist())
            }
        elif isinstance(scaler, StandardScaler):
            scaler_data.type = maingopy.melonpy.SCALER_TYPE.STANDARD          
            scaler_data.parameters = {
                maingopy.melonpy.SCALER_PARAMETER.STD_DEV : np.sqrt(scaler.var_).tolist(),
                maingopy.melonpy.SCALER_PARAMETER.MEAN : scaler.mean_.tolist()
            }
        else:
            raise Exception("Unsupported scaler type. Scaler has to be either a scikit-learn MinMaxScaler or StandardScaler instance or None (=identity(no scaling))")
        
        return scaler_data 

def generate_melon_gp_object(GP_model, X, y, matern, scaler):
    gp_data = maingopy.melonpy.GPData()
    
    gp_data.X = X.numpy()
    gp_data.Y = y.numpy()

    gp_data.nX, gp_data.DX = X.shape

    if len(y.shape) == 1:
        gp_data.DY = y.shape[0]
    else:
        gp_data.DY = y.shape[1]

    cov_mat = GP_model.covar_module(X)
    gp_data.K = cov_mat.numpy() 
    inv_cov_mat = torch.inverse(cov_mat.evaluate())
    gp_data.invK = inv_cov_mat.detach().numpy()

    gp_data.matern = matern
    kernel_data = maingopy.melonpy.KernelData()
    kernel_data.sf2 = GP_model.covar_module.outputscale.detach().numpy().astype(float).item()               #outputscale sigma*K
    kernel_data.ell = GP_model.covar_module.base_kernel.lengthscale.detach().numpy().squeeze().tolist()     #lenghtscales kernel
    gp_data.kernelData = kernel_data

    if not 'input' in scaler:
        scaler['input'] = None
    gp_data.inputScalerData = generate_melon_scaler_object(scaler['input'])


    if not 'output' in scaler or not isinstance(scaler['output'], StandardScaler):
        raise Exception('The output scaler has to be as scikit-learn StandardScaler instance')   

    gp_data.predictionScalerData = generate_melon_scaler_object(scaler['output'])

    gp_data.stdOfOutput = np.sqrt(scaler['output'].var_)[0]
    gp_data.meanFunction = 0

    return gp_data


def save_model_to_json(filepath, filename, GP_model, X, y, matern, scalers=dict()):
    prediction_parameters = dict()
    prediction_parameters["nX"] = X.shape[0]
    prediction_parameters["DX"] = X.shape[1]

    if len(y.shape) == 1:
        prediction_parameters["DY"] = y.shape[0]
    else:
        prediction_parameters["DY"] = y.shape[1]

    prediction_parameters["matern"] = matern
    prediction_parameters["meanfunction"] = 0
    prediction_parameters["X"] = X.numpy().tolist()
    prediction_parameters["Y"] = y.numpy().tolist()

    cov_mat = GP_model.covar_module(X)
    prediction_parameters["K"] = cov_mat.numpy().tolist()
    prediction_parameters["invK"] = np.linalg.inv(prediction_parameters["K"]).tolist()

    if not 'input' in scalers or not isinstance(scalers['input'], MinMaxScaler):
        raise Exception("There has to be an inputscaler which is a scikit-learn MinMaxScaler instance")

    prediction_parameters["problemLowerBound"] = scalers['input'].data_min_.tolist()
    prediction_parameters["problemUpperBound"] = scalers['input'].data_max_.tolist()
    
    if not 'output' in scalers or not isinstance(scalers['output'], StandardScaler):
        raise Exception("There has to be an output scaler which is a scikit-learn StandardScaler instance")
    
    prediction_parameters["stdOfOutput"] = np.sqrt(scalers['output'].var_).item()
    prediction_parameters["meanOfOutput"] = scalers['output'].mean_.item()

    prediction_parameters["sf2"] = GP_model.covar_module.outputscale.detach().numpy().astype(float).item()           #outputscale sigma*K
    prediction_parameters["ell"] = GP_model.covar_module.base_kernel.lengthscale.detach().numpy().squeeze().tolist()     #lenghtscales kernel

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    with open(os.path.join(filepath,filename), 'w') as outfile:
        json.dump(prediction_parameters, outfile)