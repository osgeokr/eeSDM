#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ee
import geemap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import random


# In[2]:


def plot_data_distribution(gdf, h_size=12):
    
    # Create a figure with two subplots: Yearly data distribution (left) and Monthly data distribution (right)
    plt.figure(figsize=(h_size, h_size-8))
    plt.subplot(1, 2, 1)
    
    # Calculate the counts of data for each year and sort by index
    year_counts = gdf['year'].value_counts().sort_index()
    
    # Create a bar plot for yearly data distribution
    plt.bar(year_counts.index, year_counts.values)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Yearly Data Distribution')
    
    # Display the data count within each bar
    for i, count in enumerate(year_counts.values):
        plt.text(year_counts.index[i], count, str(count), ha='center', va='bottom')
    
    # Create the second subplot for monthly data distribution
    plt.subplot(1, 2, 2)
    
    # Calculate the counts of data for each month and sort by index
    month_counts = gdf['month'].value_counts().sort_index()
    
    # Create a bar plot for monthly data distribution
    plt.bar(month_counts.index, month_counts.values)
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.title('Monthly Data Distribution')
    
    # Display the data count within each bar for the second subplot
    for i, count in enumerate(month_counts.values):
        plt.text(month_counts.index[i], count, str(count), ha='center', va='bottom')

    # Set x-axis tick labels to integer format
    plt.xticks(month_counts.index, map(int, month_counts.index))
    
    # Adjust layout for better appearance
    plt.tight_layout()
    
    # Save the plot as an image file
    plt.savefig('data_distribution_plot.png')
    
    # Display the plot
    plt.show()


# In[3]:


def plot_heatmap(gdf, h_size=8):
    # Calculate necessary statistics
    statistics = gdf.groupby(["month", "year"]).size().unstack(fill_value=0)
    
    # Visualize statistics using a heatmap
    plt.figure(figsize=(h_size, h_size-6))
    heatmap = plt.imshow(statistics.values, cmap="YlOrBr", origin="upper", aspect="auto")

    # Display values on top of each pixel
    for i in range(len(statistics.index)):
        for j in range(len(statistics.columns)):
            plt.text(j, i, statistics.values[i, j], ha="center", va="center", color="black")

    plt.colorbar(heatmap, label="Count")
    plt.title("Monthly Species Count by Year")
    plt.xlabel("Year")
    plt.ylabel("Month")
    plt.xticks(range(len(statistics.columns)), statistics.columns)
    plt.yticks(range(len(statistics.index)), statistics.index)
    plt.tight_layout()
    plt.savefig('heatmap_plot.png')
    plt.show()
    print(gdf.groupby(["month", "year"]).size().unstack(fill_value=0))


# In[4]:


def remove_duplicates(data_raw, GrainSize):
    # Select one presence record per pixel at the chosen spatial resolution (1km) randomly
    # Generate a random raster image and reproject it to the specified coordinate system and resolution
    random_raster = ee.Image.random().reproject('EPSG:4326', None, GrainSize)
    
    # Sample presence points with the generated random raster
    # Scale parameter is set to 10 for sampling, geometries are included
    rand_point_vals = random_raster.sampleRegions(collection=ee.FeatureCollection(data_raw), scale=10, geometries=True)
    
    # Keep only distinct presence records based on the 'random' property
    return rand_point_vals.distinct('random')


# In[5]:


def plot_correlation_heatmap(df, h_size=10):
    # Calculate Spearman correlation coefficients
    correlation_matrix = df.corr(method="spearman")

    # Create a heatmap
    plt.figure(figsize=(h_size, h_size-2))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')

    # Display values on the heatmap
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                     ha='center', va='center', color='white', fontsize=8)  # Adjust fontsize

    columns = df.columns.tolist()
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.title("Variables Correlation Matrix")
    plt.colorbar(label="Spearman Correlation")
    plt.savefig('correlation_heatmap_plot.png')
    plt.show()


# In[6]:


def filter_variables_by_vif(df, threshold=10):
    # Store the original column names
    original_columns = df.columns.tolist()
    
    # Create a copy of column names to track remaining variables
    remaining_columns = original_columns[:]
    
    # Perform VIF-based variable selection iteratively
    while True:
        # Create a subset of the DataFrame using remaining variables
        vif_data = df[remaining_columns]
        
        # Calculate VIF values for each remaining variable
        vif_values = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
        
        # Find the index of the variable with the highest VIF
        max_vif_index = vif_values.index(max(vif_values))
        max_vif = max(vif_values)
        
        # Check if the highest VIF is below the specified threshold
        if max_vif < threshold:
            break
        
        # Print information about the variable being removed
        print(f"Removing '{remaining_columns[max_vif_index]}' with VIF {max_vif:.2f}")
        
        # Remove the variable with the highest VIF from the list of remaining variables
        del remaining_columns[max_vif_index]
    
    # Create a new DataFrame with the filtered variables
    filtered_data = df[remaining_columns]
    bands = filtered_data.columns.tolist()
    print('Bands:', bands)
    
    return filtered_data, bands


# In[7]:


def generate_pa_full_area(Data, GrainSize, AOI):
    presence_mask = Data.reduceToImage(
        properties=['random'],
        reducer=ee.Reducer.first()
    ).reproject('EPSG:4326', None, ee.Number(GrainSize)).mask().neq(1).selfMask()
    AreaForPA = presence_mask.updateMask(ee.Image("USGS/SRTMGL1_003").gt(0)).clip(AOI)
    return AreaForPA


# In[8]:


def generate_pa_spatial_constraint(Data, GrainSize, AOI, distance=50000, maxError=1000):
    presence_mask = Data.reduceToImage(
        properties=['random'],
        reducer=ee.Reducer.first()
    ).reproject('EPSG:4326', None, ee.Number(GrainSize)).mask().neq(1).selfMask()
    presence_buffer_mask = Data.geometry().buffer(distance, maxError)
    AreaForPA = presence_mask.clip(presence_buffer_mask).updateMask(ee.Image("USGS/SRTMGL1_003").gt(0)).clip(AOI)
    return AreaForPA


# In[9]:


def generate_pa_environmental_profiling(Data, GrainSize, AOI, predictors):
    presence_mask = Data.reduceToImage(
        properties=['random'],
        reducer=ee.Reducer.first()
    ).reproject('EPSG:4326', None, ee.Number(GrainSize)).mask().neq(1).selfMask()
    # Extract predictor variable values from a random subset of the presence data
    PixelVals = predictors.sampleRegions(
        collection=Data.randomColumn().sort('random').limit(100),
        properties=[],
        tileScale=16,
        scale=GrainSize
    )
    
    # Train a k-means clustering model
    clusterer = ee.Clusterer.wekaKMeans(
        nClusters=2,
        distanceFunction="Euclidean"
    ).train(PixelVals)
    
    # Use the trained clusterer to assign pixels to clusters
    Clresult = predictors.cluster(clusterer)
    
    # Obtain cluster IDs similar to those of the presence data
    clustID = Clresult.sampleRegions(
        collection=Data.randomColumn().sort('random').limit(200),
        properties=[],
        tileScale=16,
        scale=GrainSize
    )
    
    # Use the opposite cluster, define pseudo-absence allowed area
    clustID = ee.FeatureCollection(clustID).reduceColumns(ee.Reducer.mode(), ['cluster'])
    clustID = ee.Number(clustID.get('mode')).subtract(1).abs()
    cl_mask = Clresult.select(['cluster']).eq(clustID)
    AreaForPA = presence_mask.updateMask(cl_mask).clip(AOI)
    return AreaForPA


# In[10]:


def createGrid(geometry, scale):
    # Create an image with pixel longitude and latitude vals
    lonLat = ee.Image.pixelLonLat()
    
    # Convert longitude and latitude images to integer grids
    lonGrid = lonLat.select('longitude').multiply(100000).toInt()
    latGrid = lonLat.select('latitude').multiply(100000).toInt()
    
    # Reduce grids to vectors to create the grid polygons
    rawGrid = lonGrid.multiply(latGrid).reduceToVectors(
        geometry=geometry.buffer(distance=20000, maxError=1000),
        scale=scale,
        geometryType='polygon')
    
    # Apply the grid to a raster image to obtain mean values within each grid cell
    Grid = ee.Image("USGS/SRTMGL1_003").gt(0).reduceRegions(
        collection=rawGrid,
        reducer=ee.Reducer.mean()).filter(ee.Filter.neq('mean', None))
    
    return Grid


# In[11]:


def batchSDM(Grid, Data, AreaForPA, GrainSize, bands, predictors, numiter, split=0.7, seed=None):
    # Fit SDM
    # Generate a list of random seeds
    if seed is not None:
        random.seed(seed)
    random_seeds = [random.randint(1, 1000) for _ in range(numiter)]

    def SDM(x, split):
        Seed = ee.Number(x)
    
        # Random block partition for training and validation
        GRID = ee.FeatureCollection(Grid).randomColumn(seed=Seed).sort('random')
        TrainingGrid = GRID.filter(ee.Filter.lt('random', split))  # Training grids
        TestingGrid = GRID.filter(ee.Filter.gte('random', split))  # Testing grids
        
        # Presence points
        PresencePoints = ee.FeatureCollection(Data)
        PresencePoints = PresencePoints.map(lambda feature: feature.set('PresAbs', 1))
        TrPresencePoints = PresencePoints.filter(ee.Filter.bounds(TrainingGrid))  # Training presence points
        TePresencePoints = PresencePoints.filter(ee.Filter.bounds(TestingGrid))  # Testing presence points
        
        # Pseudo-absence points
        TrPseudoAbsPoints = AreaForPA.sample(region=TrainingGrid,
                                             scale=GrainSize,
                                             numPixels=TrPresencePoints.size().add(300),
                                             seed=Seed,
                                             geometries=True)
        TrPseudoAbsPoints = TrPseudoAbsPoints.randomColumn().sort('random').limit(ee.Number(TrPresencePoints.size()))
        TrPseudoAbsPoints = TrPseudoAbsPoints.map(lambda feature: feature.set('PresAbs', 0))
        
        TePseudoAbsPoints = AreaForPA.sample(region=TestingGrid,
                                             scale=GrainSize,
                                             numPixels=TePresencePoints.size().add(100),
                                             seed=Seed,
                                             geometries=True)
        TePseudoAbsPoints = TePseudoAbsPoints.randomColumn().sort('random').limit(ee.Number(TePresencePoints.size()))
        TePseudoAbsPoints = TePseudoAbsPoints.map(lambda feature: feature.set('PresAbs', 0))
    
        # Merge training presence and pseudo-absence points
        trainingPartition = TrPresencePoints.merge(TrPseudoAbsPoints)
        testingPartition = TePresencePoints.merge(TePseudoAbsPoints)
    
        # Extract covariate values of predictor images from training partition
        trainPixelVals = predictors.sampleRegions(collection=trainingPartition,
                                                  properties=['PresAbs'],
                                                  scale=GrainSize,
                                                  tileScale=16,
                                                  geometries=True)
    
        # Random Forest classifier
        Classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=500,
            variablesPerSplit=None,
            minLeafPopulation=10,
            bagFraction=0.5,
            maxNodes=None,
            seed=Seed
        )
        
        # Presence probability classifier
        ClassifierPr = Classifier.setOutputMode('PROBABILITY').train(trainPixelVals, 'PresAbs', bands)
        ClassifiedImgPr = predictors.select(bands).classify(ClassifierPr)
        
        # Binary presence/absence classifier
        ClassifierBin = Classifier.setOutputMode('CLASSIFICATION').train(trainPixelVals, 'PresAbs', bands)
        ClassifiedImgBin = predictors.select(bands).classify(ClassifierBin)
    
        # Variable importance
        varImportance = ClassifierPr.explain().get('importance')
    
        return [varImportance, ClassifiedImgPr, ClassifiedImgBin, trainingPartition, testingPartition]
    
    # Run SDM for each seed and concatenate results
    results_list = [SDM(x, split) for x in random_seeds]
    flattened_results = ee.List(results_list).flatten()
    
    return flattened_results


# In[12]:


def plot_avg_variable_importance(results, numiter, h_size=8):
    # Calculate Average Variable Importance
    varImportance = ee.List.sequence(0, ee.Number(numiter).multiply(5).subtract(1), 5).map(lambda x: results.get(x)).getInfo()
    avg_varImportance = {key: sum(d[key] for d in varImportance) / len(varImportance) for key in varImportance[0]}

    # Sort variables by importance in descending order
    avg_varImportance = {key: value for key, value in sorted(avg_varImportance.items(), key=lambda item: item[1], reverse=False)}

    # Create a horizontal bar plot
    plt.figure(figsize=(h_size, h_size-4))  # Adjust the size as desired
    plt.barh(list(avg_varImportance.keys()), list(avg_varImportance.values()))
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.title('Average Variable Importance')

    # Display values on top of bars (rounded to two decimal places)
    for i, value in enumerate(avg_varImportance.values()):
        plt.text(value, i, f'{value:.2f}', va='center')

    # Extend x-axis range by 5 units beyond the maximum value
    max_value = max(avg_varImportance.values())
    plt.xlim(0, max_value + 5)

    plt.tight_layout()
    plt.savefig('avg_variable_importance_plot.png')
    plt.show()
    print(avg_varImportance)


# In[13]:


def print_pres_abs_sizes(TestingDatasets, numiter):
    # Checking if there are sufficient presence and pseudo-absence points
    def get_pres_abs_size(x):
        fc = ee.FeatureCollection(TestingDatasets.get(x))
        presence_size = fc.filter(ee.Filter.eq('PresAbs', 1)).size()
        pseudo_absence_size = fc.filter(ee.Filter.eq('PresAbs', 0)).size()
        return ee.List([presence_size, pseudo_absence_size])

    sizes_info = ee.List.sequence(0, ee.Number(numiter).subtract(1), 1).map(get_pres_abs_size).getInfo()
    
    for i, sizes in enumerate(sizes_info):
        presence_size = sizes[0]
        pseudo_absence_size = sizes[1]
        print(f'Iteration {i + 1}: Presence Size = {presence_size}, Pseudo-absence Size = {pseudo_absence_size}')


# In[14]:


def getAcc(HSM, TData, GrainSize):
    Pr_Prob_Vals = HSM.sampleRegions(collection=TData, properties=['PresAbs'], scale=GrainSize, tileScale=16)
    seq = ee.List.sequence(start=0, end=1, count=25) # Divide 0 to 1 into 25 intervals
    def calculate_metrics(cutoff):
        # Each element of the seq list is passed as cutoff(threshold value)
        
        # Observed present = TP + FN
        Pres = Pr_Prob_Vals.filterMetadata('PresAbs', 'equals', 1)

        # TP (True Positive)
        TP = ee.Number(Pres.filterMetadata('classification', 'greater_than', cutoff).size())
        
        # TPR (True Positive Rate) = Recall = Sensitivity = TP / (TP + FN) = TP / Observed present
        TPR = TP.divide(Pres.size())
        
        # Observed absent = FP + TN
        Abs = Pr_Prob_Vals.filterMetadata('PresAbs', 'equals', 0)
        
        # FN (False Negative)
        FN = ee.Number(Pres.filterMetadata('classification', 'less_than', cutoff).size())
        
        # TNR (True Negative Rate) = Specificity = TN  / (FP + TN) = TN / Observed absent
        TN = ee.Number(Abs.filterMetadata('classification', 'less_than', cutoff).size())
        TNR = TN.divide(Abs.size())
        
        # FP (False Positive)
        FP = ee.Number(Abs.filterMetadata('classification', 'greater_than', cutoff).size())
        
        # FPR (False Positive Rate) = FP / (FP + TN) = FP / Observed absent
        FPR = FP.divide(Abs.size())

        # Precision = TP / (TP + FP) = TP / Predicted present
        Precision = TP.divide(TP.add(FP))

        # SUMSS = SUM of Sensitivity and Specificity
        SUMSS = TPR.add(TNR)
        
        return ee.Feature(
            None,
            {
                'cutoff': cutoff,
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TPR': TPR,
                'TNR': TNR,
                'FPR': FPR,
                'Precision': Precision,
                'SUMSS': SUMSS
            }
        )
    
    return ee.FeatureCollection(seq.map(calculate_metrics))


# In[15]:


def calculate_and_print_auc_metrics(images, TestingDatasets, GrainSize, numiter):
    # Calculate AUC-ROC and AUC-PR
    def calculate_auc_metrics(x):
        HSM = ee.Image(images.get(x))
        TData = ee.FeatureCollection(TestingDatasets.get(x))
        Acc = getAcc(HSM, TData, GrainSize)

        # AUC-ROC
        X = ee.Array(Acc.aggregate_array('FPR'))
        Y = ee.Array(Acc.aggregate_array('TPR'))
        X1 = X.slice(0,1).subtract(X.slice(0,0,-1))
        Y1 = Y.slice(0,1).add(Y.slice(0,0,-1))
        auc_roc = X1.multiply(Y1).multiply(0.5).reduce('sum',[0]).abs().toList().get(0)

        # AUC-PR
        X = ee.Array(Acc.aggregate_array('TPR'))
        Y = ee.Array(Acc.aggregate_array('Precision'))
        X1 = X.slice(0,1).subtract(X.slice(0,0,-1))
        Y1 = Y.slice(0,1).add(Y.slice(0,0,-1))
        auc_pr = X1.multiply(Y1).multiply(0.5).reduce('sum',[0]).abs().toList().get(0)
        
        return (auc_roc, auc_pr)

    auc_metrics = ee.List.sequence(0, ee.Number(numiter).subtract(1), 1).map(calculate_auc_metrics).getInfo()

    # Print AUC-ROC and AUC-PR for each iteration
    df = pd.DataFrame(auc_metrics, columns=['AUC-ROC', 'AUC-PR'])
    df.index = [f'Iteration {i + 1}' for i in range(len(df))]
    df.to_csv('auc_metrics.csv', index_label='Iteration')
    print(df)

    # Calculate mean and standard deviation of AUC-ROC and AUC-PR
    mean_auc_roc, std_auc_roc = df['AUC-ROC'].mean(), df['AUC-ROC'].std()
    mean_auc_pr, std_auc_pr = df['AUC-PR'].mean(), df['AUC-PR'].std()
    print(f'Mean AUC-ROC = {mean_auc_roc:.4f} ± {std_auc_roc:.4f}')
    print(f'Mean AUC-PR = {mean_auc_pr:.4f} ± {std_auc_pr:.4f}')


# In[16]:


def calculate_and_print_ss_metrics(images, TestingDatasets, GrainSize, numiter):
    # Calculate Sensitivity and Specificity
    def calculate_ss_metrics(x):
        HSM = ee.Image(images.get(x))
        TData = ee.FeatureCollection(TestingDatasets.get(x))
        Acc = getAcc(HSM, TData, GrainSize)

        # Sensitivity
        sensitivity = Acc.sort('SUMSS', False).first().get('TPR')
        
        # Specificity
        specificity = Acc.sort('SUMSS', False).first().get('TNR')
        
        return (sensitivity, specificity)
        
    ss_metrics = ee.List.sequence(0, ee.Number(numiter).subtract(1), 1).map(calculate_ss_metrics).getInfo()

    # Print Sensitivity and Specificity for each iteration
    df = pd.DataFrame(ss_metrics, columns=['Sensitivity', 'Specificity'])
    df.index = [f'Iteration {i + 1}' for i in range(len(df))]
    df.to_csv('ss_metrics.csv', index_label='Iteration')
    print(df)

    # Calculate mean and standard deviation of Sensitivity and Specificity
    mean_sensitivity, std_sensitivity = df['Sensitivity'].mean(), df['Sensitivity'].std()
    mean_specificity, std_specificity = df['Specificity'].mean(), df['Specificity'].std()
    print(f'Mean Sensitivity = {mean_sensitivity:.4f} ± {std_sensitivity:.4f}')
    print(f'Mean Specificity = {mean_specificity:.4f} ± {std_specificity:.4f}')


# In[17]:


def plot_roc_pr_curves(images, TestingDatasets, GrainSize, numiter):
    # Plot ROC and PR curves
    def get_roc_pr_values(x):
        HSM = ee.Image(images.get(x))
        TData = ee.FeatureCollection(TestingDatasets.get(x))        
        Acc = getAcc(HSM, TData, GrainSize)

        # Get ROC and PR values
        varFPR = Acc.aggregate_array('FPR')
        varTPR = Acc.aggregate_array('TPR')
        varPrecision = Acc.aggregate_array('Precision')
        
        return (varFPR, varTPR, varPrecision)

    # Retrieve ROC and PR values for each iteration and model
    roc_pr_values = np.array(ee.List.sequence(0, ee.Number(numiter).subtract(1), 1).map(get_roc_pr_values).getInfo())

    all_model_data = []
    for model_data in roc_pr_values:
        # Transpose data to match the desired format
        transposed_data = np.transpose(model_data)
        all_model_data.append(transposed_data)
    
    # Calculate mean and standard deviation across models
    mean_data = np.mean(all_model_data, axis=0)
    std_data = np.std(all_model_data, axis=0)
    
    # Create DataFrames for mean and standard deviation data
    df_mean = pd.DataFrame(mean_data, columns=['FPR', 'TPR', 'Precision'])
    df_std = pd.DataFrame(std_data, columns=['FPR', 'TPR', 'Precision'])
    
    # Set font size and style
    plt.rcParams.update({'font.size': 14})
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 6))
    
    # Plot mean ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(df_mean['FPR'], df_mean['TPR'], color='blue', lw=2, label='Mean ROC Curve')
    plt.fill_between(df_mean['FPR'], df_mean['TPR'] - df_std['TPR'], df_mean['TPR'] + df_std['TPR'], color='gray', alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    # Plot mean PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(df_mean['TPR'], df_mean['Precision'], color='blue', lw=2, label='Mean PR Curve')
    plt.fill_between(df_mean['TPR'], df_mean['Precision'] - df_std['Precision'], df_mean['Precision'] + df_std['Precision'], color='gray', alpha=0.2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) curve')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_pr_curves_plot.png')
    plt.show()


# In[18]:


def create_DistributionMap2(images, TestingDatasets, GrainSize, numiter, ModelAverage):
    # Potential Distribution Map using the optimal threshold
    def get_metrics(x):
        HSM = ee.Image(images.get(x))
        TData = ee.FeatureCollection(TestingDatasets.get(x))
        Acc = getAcc(HSM, TData, GrainSize)
        return Acc.sort('SUMSS', False).first()
    
    metrics = ee.List.sequence(0, ee.Number(numiter).subtract(1), 1).map(get_metrics)
    MeanThresh = ee.Number(ee.FeatureCollection(metrics).aggregate_array("cutoff").reduce(ee.Reducer.mean()))
    print('Mean threshold:', MeanThresh.getInfo())
    
    DistributionMap2 = ModelAverage.gte(MeanThresh)
    
    return DistributionMap2

