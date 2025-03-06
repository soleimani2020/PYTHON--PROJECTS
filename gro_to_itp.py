import numpy as np
import sys
import typing
import pandas as pd
from colors_text import TextColor as bcolors
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
import math



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
    
    
    def __init__(self, membrane_LX: float = 43.51812, membrane_LY: float = 45.69877,  mesh_resolution: int = 1):  
        self.membrane_LX=membrane_LX
        self.membrane_LY=membrane_LY
        self.membrane_area = self.membrane_LX*self.membrane_LY
        self.mesh_resolution = mesh_resolution
        

    def _get_xy_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a mesh grid for a given membrane area."""
        mesh_size_X = self.membrane_LX / self.mesh_resolution
        mesh_size_Y = self.membrane_LY / self.mesh_resolution
        grid_area=mesh_size_X*mesh_size_Y
        
        
        x_mesh, y_mesh = np.meshgrid(
            np.arange(0.0, self.membrane_LX, mesh_size_X),
            np.arange(0.0, self.membrane_LY, mesh_size_Y)
        )

        Mesh_NUMBER=1
        

        return x_mesh, y_mesh , grid_area ,  Mesh_NUMBER , self.mesh_resolution , mesh_size_X , mesh_size_Y  , self.membrane_LX  , self.membrane_LY
    
    

    @staticmethod
    def write_gromacs_gro(gro_data: pd.DataFrame,
                          filename: str,  # Name of the output file
                          pbc_box=None,
                          title=None
                          ) -> None:
        """Write DataFrame to a GROMACS gro file."""
        
        
        df_i: pd.DataFrame = gro_data.copy()
        
        output_file_path = os.path.join(filename)
        
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
    def process_mesh(cls, x_mesh, y_mesh, mesh_size_X, mesh_size_Y, Mesh_NUMBER, mesh_resolution, xyz_i, max_z_threshold, min_z_threshold, frame):
        selected_atoms_info = {}
        selected_atoms_data_original = xyz_i
        column_names_original = ['residue_number', 'residue_name', 'atom_name', 'atom_id', 'x', 'y', 'z']
        df_selected_Raws_original = pd.DataFrame(selected_atoms_data_original, columns=column_names_original)
        #print(xyz_i[:, 4])
        x_min_mesh = 0
        x_max_mesh = 9.6
        y_min_mesh =  0
        y_max_mesh = 1000
        #mask_nc3 = np.full(len(xyz_i), True)
        #ind_in_mesh_nc3_before_mask = np.arange(len(xyz_i))
        mask_nc3 = (xyz_i[:, 2] == 'BB')
        # Apply the mask to select atoms within the current mesh element based on XY & Z
        ind_in_mesh_nc3 = np.where(
            (xyz_i[:, 4] >= x_min_mesh) &
            (xyz_i[:, 4] < x_max_mesh) &
            (xyz_i[:, 5] >= y_min_mesh) &
            (xyz_i[:, 5] < y_max_mesh) &
            (xyz_i[:, 6] < max_z_threshold) &
            (xyz_i[:, 6] > min_z_threshold) )
    
        # Select atom data based on the indices obtained
        selected_atoms_data = xyz_i[ind_in_mesh_nc3]
        #print(selected_atoms_data)
        column_names = ['residue_number', 'residue_name', 'atom_name', 'atom_id', 'x', 'y', 'z']
        df_selected_Raws = pd.DataFrame(selected_atoms_data, columns=column_names)
        #print(len(df_selected_Raws))
        #print(df_selected_Raws)
        Index_List = df_selected_Raws.iloc[:, 3].tolist()
        print(len(Index_List))

        
        # Write to file
        with open("atoms_list.txt", "w") as file:
            for atom in Index_List:
                file.write(f"{atom}\n")
        
        print("Atoms written to atoms_list.txt")



        
    
        return Index_List
        
read_gro_instance = ReadGro(fname="Myphd.gro")
gro_data = read_gro_instance.gro_data
xyz_i = gro_data[['residue_number', 'residue_name', 'atom_name', 'atom_id','x', 'y', 'z']].values
print(xyz_i)
mesh_generator = APL_ANALYSIS()
x_mesh, y_mesh , grid_area ,  Mesh_NUMBER , mesh_resolution , mesh_size_X , mesh_size_Y , membrane_LX , membrane_LY= mesh_generator._get_xy_grid()
result = mesh_generator.process_mesh(x_mesh, y_mesh, mesh_size_X, mesh_size_Y, Mesh_NUMBER, mesh_resolution,  xyz_i=xyz_i, max_z_threshold=1000, min_z_threshold=-1000, frame=0)
print(result)



