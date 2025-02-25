import MDAnalysis as mda
import numpy as np
from decimal import Decimal, getcontext
import numpy as np
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis import distances
import pandas as pd
from scipy.stats import norm
import time
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from decimal import Decimal, getcontext
from scipy import constants as cst




num_segments = 1


class Radius_Calculation:
    def __init__(self, 
                segment,
                num_segments,
                u0,  # Water MD universe
                u,   # Mem MD universe 
                atoms_positions_water=None, 
                atoms_positions_mem=None,
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs) 
        self.universe_water = u0  
        #print(self.universe_water)
        self.universe_membrane = u
        #print(self.universe_membrane)
        self.atoms_positions_water    = self.universe_water.atoms.positions
        #print("A:\n",self.atoms_positions_water)
        self.atoms_positions_membrane = self.universe_membrane.atoms.positions
        #print(self.atoms_positions_membrane)
        self.segment = segment
        self.num_segments = num_segments
        

        
    def Radius(self):

       # Select atoms from the membrane universe
        cylinder_x_min = np.min(self.universe_water.positions[:, 0])
        #print("cylinder_x_min:\n",cylinder_x_min)
        cylinder_x_max = np.max(self.universe_water.positions[:, 0])
        #print("cylinder_x_max:\n",cylinder_x_max)
        segment_y_min=0
        segment_y_max=23.63207
        segment_z_min=0
        segment_z_max=23.63207
        
        
        Total_length = (cylinder_x_max-cylinder_x_min)
        segment_width = float(Total_length / self.num_segments)
        #print("segment_width:\n",segment_width)
        
        
        segment_x_min = 0#cylinder_x_min + self.segment * segment_width
        print("segment_x_min:\n",segment_x_min)
        segment_x_max = 168.75#segment_x_min + segment_width
        print("segment_x_max:\n",segment_x_max)
        
        
        
        segment_water = self.universe_water.select_atoms(f"prop x >= {-100000000} and prop x < {100000000}")  # Water partilces in corresponding segment 
        
        # Calculate the center of mass for the segment (using lipid as the refrence)
        center_of_mass_mem = self.universe_membrane.center_of_mass()
        
        # Calculate the distances of atoms from the center of mass in the y and z directions (ignoring x)
        distances = np.linalg.norm(self.universe_membrane.positions[:, 1:3] - center_of_mass_mem[1:3], axis=1)
        
        # Calculate mean, inner, and outer radius for this segment
        MEAN_RADIUS_SEGMENT = np.mean(distances)
        MIN_RADIUS_SEGMENT  = np.min(distances)
        MAX_RADIUS_SEGMENT  = np.max(distances)
        
        # Box volume inside and outside the segment
        box_volume_inside = np.pi * (MIN_RADIUS_SEGMENT ** 2) * (segment_x_max - segment_x_min)
        box_volume_outside = ((segment_y_max - segment_y_min) * (segment_y_max - segment_z_min) * (segment_x_max - segment_x_min)) - np.pi * (MAX_RADIUS_SEGMENT ** 2) * (segment_x_max - segment_x_min)
        
        # Radius cutoff values (adjusted)
        radius_cutoff_inner = MIN_RADIUS_SEGMENT - 5  # Angstroms
        radius_cutoff_outer = MAX_RADIUS_SEGMENT + 5  # Angstroms
        
        return segment_water,center_of_mass_mem,radius_cutoff_inner, radius_cutoff_outer
        
        
## Create an instance of Radius_Calculation


#u = mda.Universe("mem_only.tpr", "mem_only.trr")
#u = u.select_atoms('all')       # Positions are in Angstroms
##print("Membrane positions:")
##print(len(u.atoms.positions))

#segment=0

#radius_calculator = Radius_Calculation(segment, u=u, atoms_positions_mem=None)
#inner_radius, outer_radius = radius_calculator.Radius()
#print("Inner radius cutoff:", inner_radius)
#print("Outer radius cutoff:", outer_radius)




#C6 = 0.21558       KJ mol^-1 nm^6
#C12 = 0.0023238    KJ mol^-1 nm^12



cut_off = 12             
epsilon=1.19503 # kcal/mol          
sigma=4.7                 
atom_mass= 72
r1=9
rc=12

r = np.arange(0.01,rc,0.001)

def get_abc(alpha,r1,rc):
    A = -((alpha+4)*rc-(alpha+1)*r1)/(rc**(alpha+2)*(rc-r1)**2)
    A = alpha*A
    B = ((alpha+3)*rc-(alpha+1)*r1)/(rc**(alpha+2)*(rc-r1)**3)
    B = alpha*B
    C = (1/rc**alpha) - (A/3)*(rc-r1)**3 - (B/4)*(rc-r1)**4
    return A,B,C

def get_s(r,alpha,r1,rc):
    A,B,C = get_abc(alpha,r1,rc)
    S = A*(r-r1)**2 + B*(r-r1)**3
    return S

def get_switched_force(r,alpha,r1,rc):
    unswitched = alpha/r**(alpha+1)
    switched = unswitched + get_s(r,alpha,r1,rc)
    switched[r<r1] = unswitched[r<r1]
    switched[rc<=r] = 0.0
    switched[r<0.04] = 0.0
    return switched


def F_vdw_switch(r):
    Fs_12= get_switched_force(r,alpha=12,r1=r1,rc=rc)
    Fs_6 = get_switched_force(r,alpha=6,r1=r1,rc=rc)
    A=  4 * epsilon * sigma**12
    C=  4 * epsilon * sigma**6
    sign = +1
    F_vdw_switch = sign * (A*Fs_12 - C*Fs_6)
    return  F_vdw_switch




      
class InitializeSimulation:
    def __init__(self, 
                 u,  # MD universe
                 atoms_positions=None,  # Array - Angstroms
                 box_size=None,  # Default to None, or replace with a specific default value
                 num_atoms=None, 
                 cutoff=None,  # Default to None, or specify a default value
                 matrix=None,  # Default to None, or specify a default value
                 p_ideal_atm=None,
                 indices =None,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)  # Ensure this is needed, else omit it

        self.universe = u  # MDAnalysis universe
        self.atoms_positions = atoms_positions
        self.box_size = box_size
        self.num_atoms = num_atoms
        self.cutoff = cutoff
        self.matrix = matrix
        self.p_ideal_atm=p_ideal_atm
        self.indices= indices
        #print("self.indices:\n",self.indices)
        #print("\n\n\n")
        
        
    def Box_Average(self):
        volume = np.prod(self.box_size[:3])   # box volume
        return volume 
        
  
                    
    def p_ideal(self):
        
        velocity = self.universe.select_atoms('name W').velocities  
        volume = np.prod(self.box_size[:3])   # box volume
        kinetic_energy =0.5* sum((atom_mass * np.dot(atom.velocity, atom.velocity)) for atom in ag)
        #print("kinetic_energy (amu)(A/ps)^2:",kinetic_energy)
        self.p_ideal = (2 * kinetic_energy) / (3 * volume)
        #print("p_ideal amu / A ps^2 \n",self. p_ideal)
        self.p_ideal_atm = self.p_ideal * 163.8
        return self.p_ideal_atm

        
        
    def calculate_pressure(self):
        """Evaluate calculate_pressure based on the Virial"""
        
        # Compute the ideal contribution
        #p_ideal_atm =  self. p_ideal_atm
        #print("p_ideal_atm:\n",p_ideal_atm)
        #print("Subset contact matrix shape in pressure:", self.matrix.shape)
        
        
        
        num_atoms = len(self.indices)
        #print("num_atoms:",num_atoms)
        

        
        MyVirial = []
        pressures = [] 
        neighbor_lists = []
        for cpt, array in enumerate(self.matrix[:-1]):
            #print(cpt)
            list = np.where(array)[0].tolist()
            list = [ele for ele in list if ele > cpt and  ele != self.indices[cpt]]
            #print("Num INT:\n",len(list))
            neighbor_lists.append(list)
            # Get positions of the current atom and its neighbors : cpt doesn't work here anymore. 
            current_atom_position = self.universe[self.indices[cpt]].position
            #print("current_atom_position in class:",current_atom_position)
            neighbor_positions = [self.universe[ele].position  for ele in list]
            distances = [np.linalg.norm(current_atom_position - neighbor_position) for neighbor_position in neighbor_positions]
            displacement = [(current_atom_position - neighbor_position) for neighbor_position in neighbor_positions]
            #print(f"Atom {self.indices[cpt]} position: {current_atom_position}")
            #print(f"Neighbors of atom {self.indices[cpt]}: {list}")
            #print(f"Neighbor positions: {neighbor_positions}")
            #print(f"displacement: {displacement}")
            #print(f"Norm of displacement vector (Before MIC): {distances}")
            #print("\n\n\n")
            
        #print("neighbor_lists:",neighbor_lists)
        
        #print("\n\n\n")
        
        

        f_vdw_switch_count = 0  # Initialize the counter
        
        total_virials_all_particles = []
        # for Ni in np.arange(np.sum(num_atoms)-1):
        for Ni in np.arange(np.sum(num_atoms)-1):
            #print("Ni:\n",Ni)
            position_of_i = self.universe[self.indices[Ni]].position
            position_of_A = position_of_i
            #print(f"position_of_A\n: {position_of_A}")
            
            
            #print(f"Atom {Ni} position: {position_of_i}")
            # Get the neighbor list for the current atom
            neighbors_of_i =  neighbor_lists[Ni]
            #print("neighbors_of_i:",neighbors_of_i)
            neighbor_positions = [self.universe[neighbor].position for neighbor in neighbors_of_i]
            #print(f"Positions of neighbors for atom {Ni}: {neighbor_positions}")
            
            # Calculate displacement vectors for each neighbor
            total_virial = 0
            for neighbor in neighbors_of_i:
                #print("neighbor:\n",neighbor)
                #print(f"position_of_A\n: {position_of_A}")
                neighbor_position = self.universe[neighbor].position
                #print(f"position_of_B\n: {neighbor_position}")
                
                ##print("neighbor_position:\n",neighbor_position)
                displacement_vector = neighbor_position - position_of_i
                norm_displacement_vector = np.linalg.norm(displacement_vector)
                #print(f"Neighbor {neighbor} position: {neighbor_position}")
                #print(f"Displacement vector to neighbor {neighbor}: {displacement_vector}")
                #print(f"Norm of displacement vector (Before MIC): {norm_displacement_vector:.10f}")
                
                #if norm_displacement_vector > 12:
                    #print("Periodic from top-botton of the cylinder!")

                

                
                ## Apply Minimum Image Convention (MIC)
                displacement_vector_mic = displacement_vector - np.round(displacement_vector / box_boundaries) * box_boundaries
                #print(f"Displacement vector to neighbor {neighbor} (after MIC): {displacement_vector_mic}")
                # Calculate norm of the MIC-adjusted displacement vector
                norm_displacement_vector_mic = np.linalg.norm(displacement_vector_mic)
                r_MIC = norm_displacement_vector_mic # Angestrum
                #print(f"Norm of displacement vector (After MIC): {norm_displacement_vector_mic:.10f}")
                
                #if r_MIC > 12:
                    #print("Sth terribely wrong !")
                    #print("Sth terribely wrong !")
                    #print("Sth terribely wrong !")
                    #print("Sth terribely wrong !")
                    #print("Sth terribely wrong !")
                    #print("Sth terribely wrong !")
                    #print("Sth terribely wrong !")
                    
                
                #print("\n")
                # Calculate potential 
                #U = potentials(epsilon, sigma, r_MIC)
                #print(f"  Pairwise potential: {U}")
                # Calculate force_ij 
                f = F_vdw_switch(np.array([r_MIC]))[0] #force(epsilon, sigma, r_MIC)  ---> kCal/mol.Angestrum
                #print(f"Pairwise Force (magnitude) between Particle 1 and Particle 2: {f:.10f}")
                f_vdw_switch_count += 1  # Increment the counter

                # Calculate Virial 
                Virial = norm_displacement_vector_mic * f # kCal/mol
                #print(f"  Virial for neighbor {neighbor}: {Virial}")
                # Accumulate virial contribution for the current atom Ni
                total_virial += Virial # kCal/mol
                #print(f"  total_virial : {total_virial}")
                #print("\n\n\n\n")
                
                

                
            ## Append total virial for this atom to the list
            total_virials_all_particles.append(total_virial)
            #print(f"Atom {Ni}: Total virial = {total_virial}") # kCal/mol
            #print("\n\n\n")
            
        # After the loop, print the count of F_vdw_switch calls
        print(f"F_vdw_switch was executed {f_vdw_switch_count} times.")
                
        ## Calculate pressure for the current frame
        #print(total_virials_all_particles)
        CONVERSION_TO_ATM = 69475.55667#4184*16.38855174 
        Sigma_Virial = (np.sum(total_virials_all_particles)) # kCal/mol
        print("Sigma_Virial:\n",Sigma_Virial)
        #print(f"Virial in kCal/mol: {0.5*(Sigma_Virial):.10f}")
        #print(f"Virial in kJ/mol: {0.5*(Sigma_Virial*4.184):.10f}")
        Sigma_Virial= 0.5*(Sigma_Virial*4.184)
        Volume = box_boundaries[0] * box_boundaries[1] * box_boundaries[2] # Angestrum^3
        #Virial = ( (np.sum(total_virials_all_particles)) / (3*Volume))   # kCal/mol.Angestrum^3 
        #print("Pressure in kcal/mol·Å³ :",Virial)
        Pressure = ( (np.sum(total_virials_all_particles)) / (3*Volume)) * CONVERSION_TO_ATM  #  atm 
        #print(f"Pressure in atm: {Virial:.10f}")

        #print("Pressure in GROMACS in atm : -0.777210")
        MyVirial.append(Sigma_Virial)
        pressures.append(Pressure)
        
        return MyVirial , pressures
                




Kinetic_values = []            
Volume_values = []
Virial_values_outside=[]
pressure_values_outside = []
Virial_values_inside=[]
pressure_values_inside = []

universe  = mda.Universe("water_only.tpr", "water_only.trr")
universe2 = mda.Universe("mem_only.tpr"  , "mem_only.trr")


for ts, ts2 in zip(universe.trajectory[:1], universe2.trajectory[:1]): 
    # Processing frames
    print(f"Processing Frame in water simulation: {ts.frame}")
    print(f"Processing Frame in membrane simulation: {ts2.frame}")
    cutoff = 12  #  cutoff in Angstroms
    box_boundaries = universe2.dimensions[:3]
    box_geometry = np.array([90, 90, 90])  # Cubic box
    box_size = np.array(box_boundaries.tolist()+box_geometry.tolist())
    
    ag_water = universe.select_atoms('all')    
    #print(ag_water.positions)
    ag_mem   = universe2.select_atoms('all') 
    #print(ag_mem.positions)
    

    

    
    
    # Loop over each segment
    for segment in range(num_segments):
   
        radius_calculator = Radius_Calculation(segment, num_segments , u0=ag_water, u=ag_mem, atoms_positions_water=None , atoms_positions_mem=None)
        segment_ag , center_of_mass ,  inner_radius, outer_radius = radius_calculator.Radius()
                

        
        selected_atoms_inside = segment_ag[[np.linalg.norm(atom.position[1:3] - center_of_mass[1:3]) < 1000000000 for atom in segment_ag]]
        #print("AAAA:",len(selected_atoms_inside))
        selected_atoms_inside_positions = np.array([atom.position for atom in selected_atoms_inside])
        #print("BBBB:",len(selected_atoms_inside_positions))
        selected_atom_inside_indices = [atom.index for atom in selected_atoms_inside] # ---> Using these indices, one can select the inside positions from the the universe (ag0) 
        #print("CCCC:",(selected_atom_inside_indices))
        selected_atoms_inside_ag = universe.atoms[selected_atom_inside_indices]
        #print("DDDD:",len(selected_atoms_inside_ag))
        selected_atoms_inside_ag_positions = np.array([atom.position for atom in selected_atoms_inside_ag])
        np.set_printoptions(precision=10, suppress=True)
        #print("EEEE:\n",len(selected_atoms_inside_ag_positions))
        #print("\n")



        #print(f"Segment {segment}:")
        print(f"- Inside cutoff: {len(selected_atoms_inside)} atoms")
        #print(f"- Outside cutoff: {len(selected_atoms_outside)} atoms")
        


                
        from MDAnalysis.analysis import distances


        # Compute the contact matrix  for sll particles in the system 
        matrix = distances.contact_matrix(
            ag_water.positions,  
            cutoff=cut_off,
            returntype="numpy",
            box=box_size  
        )
        

        # Particles in the corresponding segment & inside the cylinder 
        
        matrix = matrix[selected_atom_inside_indices,:]

        
        
        ## Check if matrix0 and matrix1 are exactly the same
        #if np.array_equal(matrix0, matrix1):
            #print("matrix0 and matrix1 are identical!")
        #else:
            #print("matrix0 and matrix1 are not identical.")
                

        
        # Transpose the positions array: converts from (N, 3) to (3, N)
        ag0 = universe.select_atoms('all') 
        num_atoms = len(ag0)
        #print("num_atoms:",len(ag0))
        P = ag0.positions   
        ag0_transposed = P.T 
        
        #print("In main:\n",ag0)


        #Create an instance of InitializeSimulation
        simulation = InitializeSimulation(
            u = ag0,                            # all particles in  (N, 3) 
            atoms_positions=ag0_transposed,     # all particles in  (3, N)
            box_size=box_boundaries,            # Box size
            num_atoms=num_atoms,                # all particles
            cutoff=cutoff,
            matrix = matrix, 
            indices = selected_atom_inside_indices,
        )
        
        
        
        #kinetic = simulation.p_ideal()
        #Kinetic_values.append(kinetic)
        Volume = simulation.Box_Average()
        Volume_values.append(Volume)
        pressure = simulation.calculate_pressure()
        Virial_values_outside.append(pressure[0])
        pressure_values_outside.append(pressure[1])
        
    All_Virial_values_outside=Virial_values_outside
    All_pressure_values_outside= pressure_values_outside
        
        




















