#include "../core/PhysiCell.h"
#include "../modules/PhysiCell_standard_modules.h"

using namespace BioFVM;
using namespace PhysiCell;

std::vector<std::vector <double>> Moment_of_Inertia_Ellipsoid(Cell* pCell, double semi_axis_x,double semi_axis_y,double semi_axis_z);

std::vector<std::vector <double>> Multiply_matrix(std::vector<std::vector <double>> M1, std::vector<std::vector <double>> M2);

std::vector<std::vector <double>> Transpose_matrix(std::vector<std::vector <double>> M);

std::vector<std::vector <double>> compute_Iinv(double semi_axis_x,double semi_axis_y,double semi_axis_z,std::vector<std::vector> rotation_matrix);


double dot_product(std::vector <double> v1,std::vector<double> v2);

double cross_product(std::vector <double> v1,std::vector<double> v2);

std::vector<double> Quaternion_multiplication(std::vector<double> quaternion_1, std::vector <double> quanternion_2);

std::vector<double> Normalize_quaternion(std::vector <double> quaternion);

std::vector<std::vector<double>> QuaternionToMatrix(std::vector<double> quaternion);

//rigid-body collision determined by using single point on object at moment of impact
//calculate the point_velocity for collisions uses world space cooridinates
std::vector<double> Point_on_object_velocity(Cell* pCell, std::vector<double> point);

double relative_velocity_magnitude(std::vector <double> point_1,Cell* pCell_1, std::vector <double> point_2, Cell* pCell_2, std::vector<double> contact_normal);

std::vector<double> Multiply_matrix_vector(std::vector<std::vector<double>> Matrix, std::vector <double> vec);

double impulse_magnitude(double COR, std::vector <double> point_1,Cell* pCell_1, std::vector <double> point_2, Cell* pCell_2, std::vector<double> contact_normal);

std::vector<double> force_from_impulse(double impulse,std::vector contact_normal);

void force_and_torque (double COR, std::vector <double> point_1,Cell* pCell_1, std::vector <double> point_2, Cell* pCell_2, std::vector<double> contact_normal);

void custom_update_velocity_from_force_and_torque(Cell* pCell, double dt);

std::vector<double> get_rotation_quaternion(double angle_in_radians, std::vector<double> axis_of_rotation);

std::vector<double> quaternion_inverse(std::vector quaternion);

std::vector<double> apply_rotation_to_cell(Cell* pCell, double angle_in_radians, std::vector<double> axis_of_rotation);
