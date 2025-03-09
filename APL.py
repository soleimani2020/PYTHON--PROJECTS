import numpy as np
import sys
import typing
import pandas as pd
import pandas as pd
import numpy as np
import sys
import typing
import os
import subprocess
from subprocess import call
import shutil
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import islice
 
NUMBER_FRAMES = 1 

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
    
    
    def __init__(self, membrane_LX: float = 43.61773, membrane_LY: float = 45.80338,  mesh_resolution: int = 2):  
        self.membrane_LX=membrane_LX
        self.membrane_LY=membrane_LY
        self.membrane_area = self.membrane_LX*self.membrane_LY
        self.mesh_resolution = mesh_resolution
        

    def _get_xy_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a mesh grid for a given membrane area."""
        mesh_size_X = self.membrane_LX / self.mesh_resolution
        #print("mesh_size_X:",mesh_size_X)
        mesh_size_Y = self.membrane_LY / self.mesh_resolution
        #print("mesh_size_Y:",mesh_size_Y)
        grid_area=mesh_size_X*mesh_size_Y
        #print("grid_area:",grid_area)
        
        
        x_mesh, y_mesh = np.meshgrid(
            np.arange(0.0, self.membrane_LX, mesh_size_X),
            np.arange(0.0, self.membrane_LY, mesh_size_Y)
        )

        L1=len(x_mesh)
        L2=len(y_mesh)
        Mesh_NUMBER=L1*L2
        print("Mesh_NUMBER:",Mesh_NUMBER)
        

        return x_mesh, y_mesh , grid_area ,  Mesh_NUMBER , self.mesh_resolution , mesh_size_X , mesh_size_Y  , self.membrane_area
    
    

    @staticmethod
    def write_gromacs_gro(gro_data: pd.DataFrame,
                          output_directory: str,
                          filename: str,  # Name of the output file
                          pbc_box=None,
                          title=None
                          ) -> None:
        """Write DataFrame to a GROMACS gro file."""
        
        
        df_i: pd.DataFrame = gro_data.copy()
        
        output_file_path = os.path.join(output_directory,filename)
        
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


    
    
    
    @classmethod
    def process_mesh(cls, x_mesh, y_mesh, mesh_size_X , mesh_size_Y , Mesh_NUMBER, mesh_resolution,  xyz_i ,max_z_threshold,min_z_threshold , frame ):        
        """Process a single frame"""
        #print("xyz_i:\n",xyz_i)
        
        
        try:
        

    
            selected_atoms_info = {}

            for i in range(x_mesh.shape[0]):
                for j in range(x_mesh.shape[1]):
                    x_min_mesh = x_mesh[i, j]
                    #print("x_min_mesh:\n",x_min_mesh)
                    x_max_mesh = x_mesh[i, j] + mesh_size_X
                    #print("x_max_mesh:\n",x_max_mesh)
                    y_min_mesh = y_mesh[i, j]
                    #print("y_min_mesh:\n",y_min_mesh)
                    y_max_mesh = y_mesh[i, j] + mesh_size_Y
                    #print("y_max_mesh:\n",y_max_mesh)

                    # Create a mask for atoms in the current mesh element where xyz_i[:, 2] == 'NC3'
                    mask_nc3 = (xyz_i[:, 2] == 'NC3')

                    # Apply the mask to select atoms within the current mesh element based on XY & Z
                    ind_in_mesh_nc3 = np.where((xyz_i[:, 4]  >= x_min_mesh) &
                                                (xyz_i[:, 4] < x_max_mesh) &
                                                (xyz_i[:, 5] >= y_min_mesh) &
                                                (xyz_i[:, 5] < y_max_mesh) &
                                                (xyz_i[:, 6] < max_z_threshold) &
                                                (xyz_i[:, 6] > min_z_threshold) &
                                                mask_nc3 )




                    grid_key = (i, j)
                    selected_atoms_info[grid_key] = []

                    for idx in ind_in_mesh_nc3[0]:
                        # Save information for the 'NC3' and  the next 9 atoms
                        selected_atoms_info[grid_key].append(xyz_i[idx])
                        for k in range(idx + 1, min(idx + 10, len(xyz_i))):
                            selected_atoms_info[grid_key].append(xyz_i[k])

                    num_atoms_in_selected_info = len(selected_atoms_info[grid_key])
                    print("num_atoms_in_selected_info:",num_atoms_in_selected_info)



            for folder_index in range(Mesh_NUMBER):
                folder_path = f"Eolder_{folder_index}"
                os.makedirs(folder_path, exist_ok=True)

            # Iterate over the selected_atoms_info dictionary and save each grid's information to a file
            for grid_key, atoms_info_list in selected_atoms_info.items():
                # Get the corresponding folder path based on the grid index
                folder_index = int(grid_key[0]) + int(grid_key[1]) * mesh_resolution
                #print(folder_index)
                folder_path = f"Eolder_{folder_index}"
                #print(folder_path)

                # Create the output file path based on grid_key
                #output_file_path = os.path.join(folder_path, f"grid_{grid_key[0]}_{grid_key[1]}_{frame}.gro")  
                output_file_path = os.path.join(folder_path, f"grid_{frame}.gro") 
                #print(output_file_path)

                # Save the data_array to the output_file_path
                data_array = np.array(atoms_info_list)
                column_names = ['residue_number', 'residue_name', 'atom_name', 'atom_id', 'x', 'y', 'z']
                df = pd.DataFrame(data_array, columns=column_names)
                #print(df)


                apl_instance = APL_ANALYSIS()
                filename="grid_"+str(grid_key[0])+"_"+str(grid_key[1])+"_"+str(frame)+".gro"
                filename="grid_"+str(frame)+".gro"
                pbc_box="43.61773  45.80338  18.04880"
                title="This file contains information about atoms in the "+str(grid_key[0])+"_"+str(grid_key[1])+" grid in frame "+str(frame)+" .\n"
                output_directory=folder_path
                apl_instance.write_gromacs_gro(df,output_directory, filename, pbc_box, title)
                
        except Exception as e:
            print(f"Empty mesh ! ")
     

        
        return 0 
    
    

    
    


def process_frame(frame):
    read_gro_instance = ReadGro(fname="conf"+str(frame)+".gro")
    gro_data = read_gro_instance.gro_data
    xyz_i = gro_data[['residue_number', 'residue_name', 'atom_name', 'atom_id','x', 'y', 'z']].values
    mesh_generator = APL_ANALYSIS()
    x_mesh, y_mesh, grid_area , Mesh_NUMBER, mesh_resolution , mesh_size_X , mesh_size_Y , membrane_area = mesh_generator._get_xy_grid()
    result = mesh_generator.process_mesh(x_mesh, y_mesh, mesh_size_X , mesh_size_Y , Mesh_NUMBER, mesh_resolution,  xyz_i=xyz_i, max_z_threshold=1000, min_z_threshold=-1000, frame=frame)
    print("End of analyzing frame "+str(frame)+" .")
    
    

if __name__ == "__main__":
    frames = range(0, NUMBER_FRAMES)
    num_processes = multiprocessing.cpu_count()  
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_frame, frames)
        
            

mesh_generator = APL_ANALYSIS()
x_mesh, y_mesh, grid_area , Mesh_NUMBER, mesh_resolution , mesh_size_X , mesh_size_Y , membrane_area = mesh_generator._get_xy_grid()



List2 = list(range(Mesh_NUMBER))
for it in range(0, len(List2)):
    NOM = List2[it]
    destination_folder = outdir + 'Eolder_' + str(NOM) + '/'
    source = indir
    shutil.copy2("APL_Post_Analysis.py", destination_folder)
    shutil.copy2("slrun_Julich_APL_Post_Analysis", destination_folder)
    shutil.copy2("Execute.sh", destination_folder)
    


def process_function(NOM):
    destination_folder = outdir + 'Eolder_' + str(NOM) + '/'
    subprocess.call('chmod +x ./Execute.sh', shell=True, cwd=destination_folder)
    subprocess.call(['bash', 'Execute.sh', str(NUMBER_FRAMES),str(mesh_size_X), str(mesh_size_Y) , str(mesh_resolution)], cwd=destination_folder)

if __name__ == '__main__':
    List2 = list(range(Mesh_NUMBER))  
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_function, List2)

    
###
###  DATA DISPLAY
###

 
My_leaflet_1=[]

# Loop over folders from Efolder_0 to Efolder_400
for i in range(Mesh_NUMBER):
    folder_name = f'Eolder_{i}'
    folder_path = os.path.join(folder_name)
    file_name = 'result_ave.txt'
    file_path = os.path.join(folder_path, file_name)
    raw_data = pd.read_csv(file_path, delim_whitespace=True, header=None, nrows=5)
    raw_data = raw_data.astype({0: 'float64'})
    My_leaflet_1.append(raw_data[0][0])

    
    
A=My_leaflet_1




def Display(input_list, layer):
    
    x, y = np.meshgrid(np.linspace(0, membrane_LX, mesh_resolution+1), np.linspace(0, membrane_LY, mesh_resolution+1))
    plt.figure(figsize=(10, 10))  


    plt.pcolormesh(x, y, np.zeros_like(x), cmap='YlGnBu')
    plt.scatter(x, y, color='black', s=2)
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))

    random_numbers = np.round(np.array(input_list), 3)
    random_numbers_grid = random_numbers.reshape(mesh_resolution, mesh_resolution)
    M=len(random_numbers_grid[0])
    print(M)
    
    # Write the first three rows to a text file
    with open('selected_values.txt', 'w') as file:
        for i in range(M):
            row = random_numbers_grid[i]
            for val in row:
                file.write(f"{val}\n")
        
        
    
    
    

    cell_size = x[0, 1] - x[0, 0]

    for j in range(mesh_resolution):
        for i in range(mesh_resolution):
            text_x = x[i, j] + cell_size / 2
            text_y = y[i, j] + cell_size / 2

            plt.text(text_x, text_y, str(random_numbers_grid[i, j]), color='black',
                    ha='center', va='center', fontsize=8)

    
    plt.savefig(f'Plot_leaflet_NUM_{layer}.png')
    plt.close()





A1=Display(A, layer=1)




 
end_time = time.time()
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60  # Convert to minutes

# Writing execution time to a text file in minutes
with open('execution_time.txt', 'w') as time_file:
    time_file.write(f"Execution time: {execution_time_minutes} minutes\n")
        

    


       

print("END OF THE CODE! GOOD LUCK2 ...")
        
        
        
 
