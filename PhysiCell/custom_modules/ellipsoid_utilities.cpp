// Author: Joseph Star Abrams
//This file contains the utilities for working with 3D ellipsoidal cell agents in physicell

//quaternion rotation: describe a cells rotatation in 3D space using the fast-to-compute quaternion q(s,v)
//requires quaternion to rotation matrix to get changes in moment of inertia tensor and quaternion multiplication
//I've also included the more common rodreguis rotation but its slower in most cases
/*
ellipsoidal cell requires custom_data:

			ellipsoidal_cell->custom_data["axis_a"];
			ellipsoidal_cell->custom_data["axis_b"];
			ellipsoidal_cell->custom_data["axis_c"];
// TODO: fill in the rest of these
Note that rigid body mechanics can be fully defined by: (Physically Based Modeling
Rigid Body Simulation, Baraff 2001)
constants:
    double mass
    matrix Inertial_tensor_of_Body (Ibody)
    matrix Ibody_inv
state variables:
    vector center
    quaternion q
    vector linear_momentum
    vector angular_momentum

auxillary variables:
    matrix Iinv (inverse of rotated interial moment) // Iinv=(rotation_matrix)(Ibody_inv)(rotation_matrix_transpose)
    matrix rotation_matrix
    vector velocity // v(t) = P(t)/mass 
    vector angular_velocity //Iinv*angular_momentum //note Iinv is 1/I and L=Iw 

computed variables:
    vector force
    vector torque

also there needes to be a COR (coefficient of restitution) for how elastic the collision is 0 is inelastic and bodies would "stick"
COR of 1 is perfectly elastic this COR can be represented as a hookes spring normalized 

*/
#include <iostream>
#include "./ellipsoid_utilities.h"
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <omp.h>
std::vector<std::vector <double>> Moment_of_Inertia_Ellipsoid(Cell* pCell, double semi_axis_x,double semi_axis_y,double semi_axis_z)
{
    //for ellipsoid in standard form going to need to keep track of what is pointing where
    std::vector<std::vector <double>> Moment;
    Moment.resize(3, std::vector<double>(3, 0.0));
    Moment[0][0]=(1/5)*pCell->custom_data["mass"]*(semi_axis_y*semi_axis_y+semi_axis_z*semi_axis_z);
    Moment[1][1]=(1/5)*pCell->custom_data["mass"]*(semi_axis_x*semi_axis_x+semi_axis_z*semi_axis_z);
    Moment[2][2]=(1/5)*pCell->custom_data["mass"]*(semi_axis_x*semi_axis_x+semi_axis_y*semi_axis_y);
}
std::vector<std::vector <double>> Multiply_matrix(std::vector<std::vector <double>> M1, std::vector<std::vector <double>> M2)
{
    std::vector<std::vector <double>> result;
    result.resize(M1.size(), std::vector<double>(M2[0].size(), 0.0));
    #pragma omp critical// not sure if needed
    {
        for(int n=0; n< M1.size();n++)
        {
            for(int i=0; i<M1.size();i++)//row M1
            {
                for(int j=0; j<M1[i].size();j++)//col M1
                {
                    result[n][i]+= M1[i][j]*M2[j][n]
                }
            }
        }
    }
    return result;
}
std::vector<std::vector <double>> Transpose_matrix(std::vector<std::vector <double>> M)
{
    std::vector<std::vector <double>> result;
    result.resize(M.size(), std::vector<double>(M[0].size(), 0.0));
    #pragma omp critical// not sure if needed
    {
        for(int i=0; i<M.size();i++)//row M1
        {
            for(int j=0; j<M[i].size();j++)//col M1
            {
                result[j][i]=M[i][j] 
            }
        }
    }
    return result;
}
std::vector<std::vector <double>> compute_Iinv(double semi_axis_x,double semi_axis_y,double semi_axis_z,std::vector<std::vector> rotation_matrix)
{
    //for ellipsoid in standard form going to need to keep track of what is pointing where
    std::vector<std::vector <double>> moment= Moment_of_Inertia_Ellipsoid(Cell* pCell, double semi_axis_x,double semi_axis_y,double semi_axis_z);
    std::vector<std::vector <double>> transpose_rotation_matrix=Transpose_matrix(rotation_matrix);
    std::vector<std::vector <double>> first_multiplication= Multiply_matrix(rotation_matrix,moment);
    std::vector<std::vector <double>> second_multiplication= Multiply_matrix(first_multiplication,transpose_rotation_matrix);
    return second_multiplication;
}

double dot_product(std::vector <double> v1,std::vector<double> v2)
{
    value=0.0;
    if(v1.size()!=v2.size())
    {
        std::cout<<" WARNING!! vector sizes don't match dot product is 0.0"<<"\n";
        return value;
    }
    else
    {
        #pragma omp for reduction(+:value)
        {
            for(int i=0; i< v1.size(); i++)
            {
                value+=v1[i]*v2[i]
            }
        }
        return value;
    }
}
double cross_product(std::vector <double> v1,std::vector<double> v2)
{
    std::vector<double> value(3,0.0);
    if(v1.size()!=v2.size())
    {
        std::cout<<" WARNING!! vector sizes don't match cross product is 0-vector"<<"\n";
        return value;
    }
    else
    {
        value[0]=v1[1]v2[2]-v1[2]v2[1];
        value[1]=v1[2]v2[0]-v1[0]v2[2];
        value[2]=v1[0]v2[1]-v1[1]v2[0];
        return value;
    }
}
std::vector<double> Quaternion_multiplication(std::vector<double> quaternion_1, std::vector <double> quanternion_2)
{
    //q=[s,v]=s+vi+vj+vk
    //[s1, v1][s2 , v2 ] = [s1 s2 − v1 · v2, s1 v2 + s2 v1 + v1 × v2 ].
    std::vector<double> product(4,0.0);
    std::vector<double> vec_part1={quaternion_1[1],quaternon_1[2],quaternion_1[3]};
    std::vector<double> vec_part2={quaternion_2[1],quaternon_2[2],quaternion_2[3]};
    std::vecotr<double> final_vec_part;
    product[0]=quaternion[0]*quaternion[0]-dot_product(quaternion_1,quaternion_2);
    final_vec_part=(quaternion_1[0]*vec_part2)+(quaternion_2[0]*vec_part1)+cross_product(vec_part1,vec_part2);
    for (int i=1; i<product.size(); i++)
    {
        product[i]=final_vec_part[i];
    }
    return product;
}
std::vector<double> Normalize_quaternion(std::vector <double> quaternion)
{
    std::vector <double> normalized_quaternion(4,0.0);
    //denominator=q[0]**2+q[1]**2+q[2]**2+q[3]**2
    double denominator= (quaternion[0]*quaternion[0])+(quaternion[1]*quaternion[1])+(quaternion[2]*quaternion[2])+(quaternion[3]*quaternion[3]);
    normalized_quaternion= quaternion/denominator;    
    return normalized_quaternion;
}
std::vector<std::vector<double>> QuaternionToMatrix(std::vector<double> quaternion)
{
    //NOTE!!! must be normalized for calculating rotation matrix
    std::vector normal_q(4,0.0);
    std::vector<std::vector<double>> Matrix;
    normal_q=Normalize_quaternion(quaternion);
    //m rows x n cols matrix[m][n]
    Matrix[0][0]=1-(2*(normal_q[2]*normal_q[2]))-(2*(normal_q[3]*normal_q[3]));
    Matrix[0][1]= (2*(normal_q[1]*normal_q[2]))-(2*(normal_q[0]*normal_q[3]));//2vxvy-2svz
    Matrix[0][2]= (2*(normal_q[1]*normal_q[3]))+(2*(normal_q[0]*normal_q[2]));
    Matrix[1][0]=(2*(normal_q[1]*normal_q[2]))+(2*(normal_q[0]*normal_q[3]));
    Matrix[1][1]=1-(2*(normal_q[1]*normal_q[1]))-(2*(normal_q[3]*normal_q[3]));
    Matrix[1][2]=(2*(normal_q[2]*normal_q[3]))-(2*(normal_q[0]*normal_q[1]));
    Matrix[2][0]=(2*(normal_q[1]*normal_q[3]))-(2*(normal_q[0]*normal_q[2]));
    Matrix[2][1]=(2*(normal_q[2]*normal_q[3]))+(2*(normal_q[0]*normal_q[1]));
    Matrix[2][2]=1-(2*(normal_q[1]*normal_q[1]))-(2*(normal_q[2]*normal_q[2]));
    return Matrix;
}
//rigid-body collision determined by using single point on object at moment of impact
//calculate the point_velocity for collisions uses world space cooridinates
std::vector<double> Point_on_object_velocity(Cell* pCell, std::vector<double> point)
{
    // might be good to have error checking to make sure point is on/in the pCell
    std::vector<double> R_vec(3,0.0);
    std::vector<double> angular_velocity(3,0.0);
    R_vec=point-pCell-position;
    //R_vec[0]=(pCell->custom_data["collision_point_x"]-pCell->position[0]); 
    //R_vec[1]=(pCell->custom_data["collision_point_y"]-pCell->position[1]);
    //R_vec[2]=(pCell->custom_data["collision_point_z"]-pCell->position[2]);

//    pCell->custom_data["R_vec_x"]=R_vec[0]; 
//    pCell->custom_data["R_vec_y"]=R_vec[1];
//    pCell->custom_data["R_vec_z"]=R_vec[2];

    angular_velocity[0]=pCell->custom_data["angular_velocity_x"]; 
    angular_velocity[1]=pCell->custom_data["angular_velocity_y"];
    angular_velocity[2]=pCell->custom_data["angular_velocity_z"];
    

    std::vector<double> d_velocity_dt(3,0.0);
    d_velocity_dt=pCell->velocity+cross_product(angular_velocity,R_vec);
    return d_velocity_dt;
}
double relative_velocity_magnitude(std::vector <double> point_1,Cell* pCell_1, std::vector <double> point_2, Cell* pCell_2, std::vector<double> contact_normal)
{
    //note positive is objects moving away from each other, should be negative for collision or less than some epsilon for "normal contact"
    std::vector <double> p1_velocity(3,0.0);
    p1_velocity=Point_on_object_velocity(pCell_1,point_1);
    std::vector <double> p2_velocity(3,0.0);
    p2_velocity=Point_on_object_velocity(pCell_2,point_2);
    double result=dot_product(contact_normal, (p1_velocity-p2_velocity));
    return result;
}
std::vector<double> Multiply_matrix_vector(std::vector<std::vector<double>> Matrix, std::vector <double> vec)
{
    //normal Matrix A multiplied by vector B in the form AB I assume they put in the right sized stuff
    //Hardcoded to avoid hijinx but probably not needed and could be operator overloaded but again not needed
   
        std::vector <double> result(3,0.0);
        result[0]=Matrix[0][0]*vec[0]+ Matrix[0][1]*vec[1]+Matrix[0][2]*vec[2];
        result[1]=Matrix[1][0]*vec[0]+ Matrix[1][1]*vec[1]+Matrix[1][2]*vec[2];
        result[2]=Matrix[2][0]*vec[0]+ Matrix[2][1]*vec[1]+Matrix[2][2]*vec[2];
}
double impulse_magnitude(double COR, std::vector <double> point_1,Cell* pCell_1, std::vector <double> point_2, Cell* pCell_2, std::vector<double> contact_normal)
{
    double impulse=0.0;
    if(COR>1 or COR<0)
    {
        std::cout<<"Bad COR returning 0 impulse"<< "\n";
        return impulse;
    }
    else
    {
        //code based on Baraff 2001
        //wrote out all the steps but can probably be compacted to gain speed, maybe the compiler optimization already does this?
        std::vector <double> R1(3,0.0);
        R1=point_1-pCell_1->position;
        std::vector <double> R2(3,0.0);
        R2=point_2-pCell_2->position;
        double vrel=relative_velocity_magnitude(point_1,pCell_1, point_2, pCell_2, contact_normal);
        double numerator= -1*(1+COR)*vrel;
        double partial_denominator_1= 1/pCell_1->custom_data["mass"];
        double partial_denominator_2= 1/pCell_2->custom_data["mass"];
        std::vector<double> R1_single_cross=cross_product(R1,contact_normal);
        std::vector<double> R1_inertia=Multiply_matrix_vector(Iinv_1,R1_single_cross);
        std::vector<double> R1_double_cross=cross_product(R1_inertia,R1);
        double partial_denominator_3=dot_product(contact_normal, R1_double_cross);
        std::vector<double> R2_single_cross=cross_product(R2,contact_normal);
        std::vector<double> R2_inertia=Multiply_matrix_vector(Iinv_2,R2_single_cross);
        std::vector<double> R2_double_cross=cross_product(R2_inertia,R2);
        double partial_denominator_4=dot_product(contact_normal, R2_double_cross);
        impulse=numerator/(partial_denominator_1+partial_denominator_2+partial_denominator_3+partial_denominator_4);
        return impulse;
    }
}
std::vector<double> force_from_impulse(double impulse,std::vector contact_normal)
{
    std::vector force(3,0.0);
    force=impulse*contact_normal;
    return force;
}
void force_and_torque (double COR, std::vector <double> point_1,Cell* pCell_1, std::vector <double> point_2, Cell* pCell_2, std::vector<double> contact_normal)
{
    //in theory interpentration requires calculating collisions are multiple points to account for how far/how much the
    // pentration has occured, I think 2 points could be used for most ellipsoidal collisions with interpentration, 
    //the two points I suggest are the deepest part of each ellipsoid.
    //contact normal points out from pCell_2
    double j = impulse_magnitude(COR, point_1,pCell_1, point_2, pCell_2,contact_normal);
    // linear momentum update for pCells force in direction of contact normal out from pCell_2 ()
    pCell_1->custom_data["linear_momentum_x"]=pCell_1->custom_data["linear_momentum_x"]+force[0];
    pCell_1->custom_data["linear_momentum_y"]=pCell_1->custom_data["linear_momentum_y"]+force[1];
    pCell_1->custom_data["linear_momentum_z"]=pCell_1->custom_data["linear_momentum_z"]+force[2];
    pCell_2->custom_data["linear_momentum_x"]=pCell_2->custom_data["linear_momentum_x"]-force[0];
    pCell_2->custom_data["linear_momentum_y"]=pCell_2->custom_data["linear_momentum_y"]-force[1];
    pCell_2->custom_data["linear_momentum_z"]=pCell_2->custom_data["linear_momentum_z"]-force[2];
    
    std::vector<double> R1_cross=cross_product(R1,force);
    pCell_1->custom_data["angular_momentum_x"]=pCell_1->custom_data["angular_momentum_x"]+R1_cross[0];
    pCell_1->custom_data["angular_momentum_y"]=pCell_1->custom_data["angular_momentum_y"]+R1_cross[1];
    pCell_1->custom_data["angular_momentum_z"]=pCell_1->custom_data["angular_momentum_z"]+R1_cross[2];

    std::vector<double> R2_cross=cross_product(R2,force);
    pCell_2->custom_data["angular_momentum_x"]=pCell_2->custom_data["angular_momentum_x"]+R2_cross[0];
    pCell_2->custom_data["angular_momentum_y"]=pCell_2->custom_data["angular_momentum_y"]+R2_cross[1];
    pCell_2->custom_data["angular_momentum_z"]=pCell_2->custom_data["angular_momentum_z"]+R2_cross[2];
// continue here: c->a->omega = c->a->Iinv * c->a->L;
}
void custom_update_velocity_from_force_and_torque(Cell* pCell, double dt)
{
    //linear velocity
    pCell->velocity[0]= pCell->custom_data["linear_momentum_x"]/pCell->custom_data["mass"];
    pCell->velocity[1]= pCell->custom_data["linear_momentum_y"]/pCell->custom_data["mass"];
    pCell->velocity[2]= pCell->custom_data["linear_momentum_z"]/pCell->custom_data["mass"];
    //angular velocity
    std::vector<double> angular_momentum(3,0.0);
    angular_momentum[0]=pCell->custom_data["angular_momentum_x"];
    angular_momentum[1]=pCell->custom_data["angular_momentum_y"];
    angular_momentum[2]=pCell->custom_data["angular_momentum_z"];

    std::vector<double> angular_velocity=Multiply_matrix_vector(Iinv,angular_momentum); 
    pCell->custom_data["angular_velocity_x"]=angular_velocity[0];
    pCell->custom_data["angular_velocity_y"]=angular_velocity[1];
    pCell->custom_data["angular_velocity_z"]=angular_velocity[2];

}
std::vector<double> get_rotation_quaternion(double angle_in_radians, std::vector<double> axis_of_rotation)
{
    double sin_part=sin(angle_in_radians/2);
    double cos_part=cos(angle_in_radians/2);
    std::vector <double> rotation_quaternion(4,0.0);
    for(int i=1; i<rotation_quaternion.size();i++)
    {
        //vector part
        rotation_quaternion[i]=axis_of_rotation[i-1]*sin_part;
    }
    rotation_quaternion[0]=cos_part;
    return rotation_quaternion;
}
std::vector<double> quaternion_inverse(std::vector quaternion)
{
    std::vector <double> result(4,0.0);
    std::vector <double> q_conj=-1*quaternion;
    result=q_conj/norm(quaternion);
    return result;
}
std::vector<double> apply_rotation_to_cell(Cell* pCell, double angle_in_radians, std::vector<double> axis_of_rotation)
{
    std::vector<double> rotation_quaternion = get_rotation_quaternion(angle_in_radians, axis_of_rotation);
    for(int i=0; i<orientation.size();i++)
    {
        //make into pure quaternion
        q_orientation[i+1]=pCell->orientation[i]
    }
    std::vector<double> new_q(4,0.0);
    std::vector<double> inverse_rotation_quaternion(4,0.0);
    inverse_rotation_quaternion=quaternion_invers(rotation_quaternion);
    new_q= Quaternion_multiplication(q_orientation, inverse_rotation_quaternion);
    new_q= Quaternion_multiplication(rotation_quaternion,new_orientation);
    for(int i=0; i<orientation.size();i++)
    {
        //make into pure quaternion
        pCell->orientation[i]=new_q[i+1];
    }
    
}