"""
Utils file for Drosophila experiments
"""

import pandas as pd
import numpy as np
import logging
import torch
import csv
from tqdm import tqdm
import anndata as ad
import scipy.sparse as sp
from scipy.io import mmwrite
from io import StringIO

"""
Given seurat clusters and timepoints, filters big metadata file for cells

Parameters:
df: big metadata file for cells (columns: cell_ids, seurat clusters, timepoints, measurements, etc.)
- for drosophila: df = pd.read_csv('/Users/jc/Documents/GitHub/Sagittarius/models/meta.csv')

selected_clusters: seurat clusters to select samples from
- in previous experiment with 100 cell IDs: [0, 1, 2, 3, 4]

selected_time_points: time windows for experiment
- in previous experiment: ['hrs_00_02', 'hrs_01_03', 'hrs_02_04', 'hrs_03_07', 'hrs_04_08']


samples_per_time_point: number of samples to take from each time point under each seurat cluster
- in previous experiment: 4

output_file: name of output
- old: '935117.csv'
"""
def filter_by_seurat_cluster(df, selected_clusters, selected_time_points, samples_per_time_point, output_file, cell_id):

    filtered_df = df[
        (df['seurat_clusters'].isin(selected_clusters)) &
        (df['time'].isin(selected_time_points))
    ].groupby(['seurat_clusters', 'time']).head(samples_per_time_point)

    filtered_df.reset_index(drop=True, inplace=True)

    filtered_df.to_csv(output_file, index=False)





"""
Given a collection of desired cell IDs based on metadata annotations, filters expr for pairs 
of measurements: (gene, cell), measurement

Parameters:
columns_metadata: cell annotations, GSE190147_scirnaseq_gene_matrix.columns.csv

rows_metadata: gene annotations, GSE190147_scirnaseq_gene_matrix.rows.csv

gene_expression_matrix: paired data for genes and cells measurements, GSE190147_scirnaseq_gene_matrix.txt

df: filtered big metadata for cells as created by filter_by_seurat_cluster

output_file: name of output
- old: 'test_25_filtered_matrix.csv'
"""
def filter_expr(columns_metadata, gene_expression_matrix, df, output_file):
    # create set of wanted cell identifiers
    cell_types = set()
    for cell_value in df['cell']:
        cell_types.add(cell_value)

    first_column_values = gene_expression_matrix.iloc[1:, 0].to_numpy()
    second_column_values = gene_expression_matrix.iloc[1:, 1].to_numpy()
    third_column_values = gene_expression_matrix.iloc[1:, 2].to_numpy()

    adata = ad.AnnData()
    data_matrix = np.column_stack((first_column_values, second_column_values, third_column_values))
    adata = ad.AnnData(data_matrix)

    adata.var_names = ['First_Column', 'Second_Column', 'Third_Column']


    wanted_values = set()
    for index, cell_id in enumerate(columns_metadata['cell']):
        if cell_id in cell_types:
            wanted_values.add(index)

    second_column = adata[:, 1].X.flatten()
    indices_to_keep = [i for i, value in enumerate(second_column) if value in wanted_values]

    filtered_adata = adata[indices_to_keep, :]

    selected_df = pd.DataFrame(data=filtered_adata.X, columns=['First_Column', 'Second_Column', 'Third_Column'])
    selected_df.to_csv(output_file, index=False)


"""
Creates file with names of all genes in expr for training

Parameters: 
filtered_from_ordered_meta: filtered expr from filter_expr()

rows_metadata: gene annotations GSE190147_scirnaseq_gene_matrix.rows.csv

output_file: name of output
- old: "gene_names_100.csv"
"""

"""
def create_genes_list(filtered_from_ordered_meta, rows_metadata, output_file, header):
    result_data = []
    with open(filtered_from_ordered_meta, "r") as matrix_file:
        csv_reader = csv.reader(matrix_file)
        if header:
            next(csv_reader)

        for line in tqdm(csv_reader, desc="Processing"):
            try:
                row_idx, col_idx, value = map(int, line)
                        
            except ValueError as ve:
                logging.error(f"Error parsing line: {line.strip()} - {ve}")
                continue
            gene_name = rows_metadata.iloc[row_idx - 2, 0]
            result_data.append([gene_name])
            
    result_df = pd.DataFrame(result_data, columns=['gene_name'])

    result_df.to_csv(output_file, index=False)
"""


def create_genes_list(filtered_from_ordered_meta, rows_metadata, output_file):
    result_data = []
    with open(filtered_from_ordered_meta, "r") as matrix_file:
        csv_reader = matrix_file
        next(csv_reader)

        for line in csv_reader:
            values = line.split()
            try:
                row_idx, col_idx, value = map(int, values)
                
                        
            except ValueError as ve:
                logging.error(f"Error parsing line: {line.strip()} - {ve}")
                continue
            gene_name = rows_metadata.iloc[row_idx - 2, 0]
            result_data.append([gene_name])
            
    result_df = pd.DataFrame(result_data, columns=['gene_name'])

    result_df.to_csv(output_file, index=False)



"""
Creates permanent mapping for each gene to index in expr for training
Parameters: 
gene_names: a list of names of genes as created by create_genes_list()
- old gene_names: "gene_names_100.csv"

output_file: name of output
- old: "gene_types_dict.txt"
"""
def create_gene_dict(gene_names, output_file):
    gene_types = {}
    counter = 0
    with open(gene_names, newline='') as gene_file:
        csv_reader = csv.reader(gene_file)
        next(csv_reader, None)
        for index, row in enumerate(csv_reader):
            if index == 0: # ensure csv indexing is correct
                print(row[0])
            gene_name = row[0]
            
            if (gene_name not in gene_types):
                gene_types[gene_name] = counter # this should be 0-based indexing
                counter += 1

    gene_types = {key: int(value) for key, value in gene_types.items()}
    with open(output_file, 'w') as file:
        for key, value in gene_types.items():
            file.write(f'{key}: {value}\n')

"""
Creates mapping between cell ids and seurat cluster for ys tensor
Parameters:

"""
def map_cell_id(shortened_meta, output_file):
    df = pd.read_csv(shortened_meta)
    cell_types = set()
    for cell_value in df['cell']:
        cell_types.add(cell_value)
    cell_type_mapping = {}
    
    with open(shortened_meta, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
    
        for row in csv_reader:
            cell = row['cell']
            seurat_cluster = row['seurat_clusters']
    
            cell_type_mapping[cell] = int(seurat_cluster)
    
    with open(output_file, 'w') as file:
        for key, value in cell_type_mapping.items():
            file.write(f'{key}: {value}\n')
    

