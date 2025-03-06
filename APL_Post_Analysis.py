import multiprocessing
import pandas as pd
import numpy as np
import sys
import typing
import pandas as pd
import os
import math
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os import path
from   itertools import islice
import sys
import statistics
from scipy.stats import norm
import matplotlib.pyplot as plt
from statistics import mean




NUMBER_FRAMES=int(sys.argv[1])
print("NUMBER_FRAMES in ANALYSIS part:",NUMBER_FRAMES)
mesh_size_X=float(sys.argv[2])
print("mesh_size_X in ANALYSIS part:",mesh_size_X)
mesh_size_Y=float(sys.argv[3])
print("mesh_size_Y in ANALYSIS part:",mesh_size_Y)
grid_area   = mesh_size_X*mesh_size_Y
print("grid_area in ANALYSIS part:",grid_area)
mesh_resolution   = int(sys.argv[4])
print("mesh_resolution in ANALYSIS part:",mesh_resolution)





class ReadGro:
    """reading GRO file based on the doc"""

    info_msg: str = 'Message:\n'  # Message to pass for logging and writing
    line_len: int = 45  # Length of the lines in the data file
    gro_data: pd.DataFrame  # All the informations in the file
    # The follwings will set in __process_header_tail method:
    title: str  # Name of the system
    number_atoms: int  # Total number of atoms in the system
    pbc_box: str  # Size of the box (its 3 floats but save as a string)

    def __init__(self,
                 fname: str,  # Name of the input file
                 ) -> None:
        self.gro_data = self.read_gro(fname)

    def read_gro(self,
                 fname: str  # gro file name
                 ) -> pd.DataFrame:
        """read gro file lien by line"""
        counter: int = 0  # To count number of lines
        processed_line: list[dict[str, typing.Any]] = []  # All proccesed lines
        with open(fname, 'r', encoding='utf8') as f_r:
            while True:
                line = f_r.readline()
                if len(line) != self.line_len:
                    self.__process_header_tail(line.strip(), counter)
                else:
                    processed_line.append(self.__process_line(line.rstrip()))
                counter += 1
                if not line.strip():
                    break
        ReadGro.info_msg += f'\tFile name is {fname}\n'        
        ReadGro.info_msg += f'\tSystem title is {self.title}\n'
        ReadGro.info_msg += f'\tNumber of atoms is {self.number_atoms}\n'
        ReadGro.info_msg += f'\tBox boundary is {self.pbc_box}\n'
        return pd.DataFrame(processed_line)

    @staticmethod
    def __process_line(line: str  # Data line
                       ) -> dict[str, typing.Any]:
        """process lines of information"""
        resnr = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnr = int(line[15:20])
        a_x = float(line[20:28])
        a_y = float(line[28:36])
        a_z = float(line[36:44])
        processed_line: dict[str, typing.Any] = {
                                                 'residue_number': resnr,
                                                 'residue_name': resname,
                                                 'atom_name': atomname,
                                                 'atom_id': atomnr,
                                                 'x': a_x,
                                                 'y': a_y,
                                                 'z': a_z,
                                                }
        return processed_line

    def __process_header_tail(self,
                              line: str,  # Line in header or tail
                              counter: int  # Line number
                              ) -> None:
        """Get the header, number of atoms, and box size"""
        if counter == 0:
            self.title = line
        elif counter == 1:
            self.number_atoms = int(line)
        elif counter == self.number_atoms + 2:
            self.pbc_box = line







class APL_ANALYSIS:
    
    
    def __init__(self, mesh_size_X: float = mesh_size_X, mesh_size_Y: float = mesh_size_Y, mesh_resolution: int = mesh_resolution):
        self.mesh_size_X = mesh_size_X
        self.mesh_size_Y = mesh_size_Y
        self.mesh_resolution = mesh_resolution



    @staticmethod
    def write_gromacs_gro(gro_data: pd.DataFrame,
                          output_directory: str,
                          filename: str,  # Name of the output file
                          pbc_box=None,
                          title=None
                          ) -> None:
        """Write DataFrame to a GROMACS gro file."""
        
        
        df_i: pd.DataFrame = gro_data.copy()
        
        output_file_path = os.path.join(output_directory, filename)
        
        with open(output_file_path, 'w', encoding='utf8') as gro_file:
            if title:
                gro_file.write(f'{title}')  # Add a comment line
            gro_file.write(f'{len(df_i)}\n')  # Write the total number of atoms
            for _, row in df_i.iterrows():
                line = f'{row["residue_number"]:>5}' \
                       f'{row["residue_name"]:<5}' \
                       f'{row["atom_name"]:>5}' \
                       f'{row["atom_id"]:>5}' \
                       f'{row["x"]:8.3f}' \
                       f'{row["y"]:8.3f}' \
                       f'{row["z"]:8.3f}\n'
                gro_file.write(line)
            if pbc_box:
                gro_file.write(f'{pbc_box}\n')
                
                
                
    @staticmethod
    def Lipid_vector(P,Q):
        POINTS = [P,Q]
        P, Q  = POINTS
        px, py, pz = P
        qx, qy, qz = Q
        distance =[px -qx, py - qy ,pz-qz ]
        norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2+distance[2] ** 2)
        direction = [distance[0] / norm, distance[1] / norm,distance[2] / norm]
        return direction
    
    
    @staticmethod 
    def Angle(vector_1,vector_2):
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle_rad = np.arccos(dot_product)
        deg = np.rad2deg(angle_rad)
        return deg 
    
    

    @staticmethod
    def perform_cluster_analysis(df, num_clusters=1):
        X = df[['z']]
        num_clusters = 1  # Replace with your desired number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        #print("df cluster:\n")
        #print(df)
        
        
        #fig = plt.figure()        
        #ax = fig.add_subplot(111, projection='3d')
        #scatter = ax.scatter(df['x'], df['y'], df['z'], c=df['cluster'], cmap='viridis')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #ax.set_title('Cluster Analysis in 3D')
        #fig.colorbar(scatter, ax=ax, label='Cluster')
        #plt.show()
        

        
        cluster_counts = df['cluster'].value_counts()
        # Compute the mean 'z' value for each cluster
        cluster_means = df.groupby('cluster')['z'].mean()
        result_df = pd.DataFrame({'Cluster': cluster_counts.index, 'Count': cluster_counts.values, 'Mean z': cluster_means.values})
        df = result_df.sort_values(by='Mean z')
        df = result_df.sort_values(by='Mean z').reset_index(drop=True)
        #print(df)
        NUM_LIPIDS_LEAFLET1=df.loc[0 , 'Count']  # Down leaflet
        #print(NUM_LIPIDS_LEAFLET1)


        
        return NUM_LIPIDS_LEAFLET1 
        

    @classmethod
    def process_mesh(cls,xyz_i, frame ):        
        """Process a single frame"""
        data_array = np.array(xyz_i)
        column_names = ['residue_number', 'residue_name', 'atom_name', 'atom_id', 'x', 'y', 'z']
        df = pd.DataFrame(data_array, columns=column_names)
        #print(df)
        raw_data_oct1 = df.astype({'residue_number':'str', 'residue_name':'str', 'atom_name':'str', 'atom_id':'str', 'x':'float64', 'y':'float64', 'z':'float64'})
        N=len(raw_data_oct1)
        #print(N)
        residue_number=raw_data_oct1['residue_number']
        residue_name=raw_data_oct1['residue_name']
        atom_name=raw_data_oct1['atom_name']
        atom_id=raw_data_oct1['atom_id']
        X=raw_data_oct1['x']
        Y=raw_data_oct1['y']
        Z=raw_data_oct1['z']

        
        Leaf_let1_3_H=[]
        
        
        for i in range(0,len(residue_number)):
            if (raw_data_oct1['atom_name'][i] == "NC3" or raw_data_oct1['atom_name'][i] == "PO4" or raw_data_oct1['atom_name'][i] == "GL1" or raw_data_oct1['atom_name'][i] == "GL2" or raw_data_oct1['atom_name'][i] == "C1A" or raw_data_oct1['atom_name'][i] == "C2A" or raw_data_oct1['atom_name'][i] == "C3A" or raw_data_oct1['atom_name'][i] == "C1B" or raw_data_oct1['atom_name'][i] == "C2B" or raw_data_oct1['atom_name'][i] == "C3B"):
        
                Trj_NC3=[raw_data_oct1['x'][i],raw_data_oct1['y'][i],raw_data_oct1['z'][i]]
                Leaf_let1_3_H.append(Trj_NC3)

                    
                    
                    
        My_Leaf_let1_3_H=Leaf_let1_3_H
        #print("My_Leaf_let1_3_H:")
        #print(My_Leaf_let1_3_H)
        df_Leaf_let1_3_H = pd.DataFrame(My_Leaf_let1_3_H, columns=['x', 'y', 'z'])
        Cluster_instance = APL_ANALYSIS()
        A=Cluster_instance.perform_cluster_analysis(df_Leaf_let1_3_H,1)

        
        
        return A


  
    

try:
    output_file_path = 'results.txt'
    with open(output_file_path, 'w') as file:
        for frame in range(NUMBER_FRAMES):
            fname = f"grid_{frame}.gro"
            if os.path.exists(fname):  # Check if the file exists
                print(fname)
                read_gro_instance = ReadGro(fname=fname)
                gro_data = read_gro_instance.gro_data
                xyz_i = gro_data[['residue_number', 'residue_name', 'atom_name', 'atom_id','x', 'y', 'z']].values
                mesh_generator = APL_ANALYSIS()
                result = mesh_generator.process_mesh(xyz_i=xyz_i, frame=frame)
            else:
                result = 0  # If file doesn't exist, set result to 0
            file.write(f'{result}\n')

except Exception as e:
    print(f"An error occurred: {e}")



import pandas as pd

# Read the text file into a DataFrame
df = pd.read_csv("results.txt", header=None)

# Calculate the mean of the column

#mean_value = int(df.mean().values[0])
mean_value = df.mean().values[0]

# Write the mean to a text file
with open("result_ave.txt", "w") as result_file:
    result_file.write(str(mean_value))

print("Mean of the column written to result_ave.txt''")




print("END OF THE CODE! GOOD LUCK ...")

































