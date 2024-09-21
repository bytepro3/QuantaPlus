#include <Eigen/Dense>
#include <random>
#include <vector>
#include"../liquid/liquid.h"
#include <omp.h>  // Include OpenMP header
#include <iomanip>
#include <chrono>  // Include for time-based seeding
#include <unsupported/Eigen/MatrixFunctions>
using namespace LiQuID;
using namespace std::complex_literals; //needed to use the literal imaginary unit [ 1i = (0,1)] 

/*Eigen::VectorXcd evolve_under_H_eff(const Eigen::MatrixXcd& H_eff, const Eigen::VectorXcd& psi, double dt) {
    // Perform the Euler time evolution step
    return (Eigen::MatrixXcd::Identity(psi.size(), psi.size()) - std::complex<double>(0, 1) * H_eff * dt) * psi;
}*/



Eigen::VectorXcd evolve_under_H_eff(const Eigen::MatrixXcd& H_eff, const Eigen::VectorXcd& psi, double dt) {
    // Compute the matrix exponential exp(-i H_eff * dt)
    Eigen::MatrixXcd U = (-std::complex<double>(0, 1) * H_eff * dt).exp();
    // Apply the evolution to the state
    return U * psi;
}
/*
Eigen::VectorXcd evolve_under_H_eff(const Eigen::MatrixXcd& H_eff, const Eigen::VectorXcd& psi, double dt) {
    // Define the time evolution function
    auto f = [&](const Eigen::VectorXcd& state) {
        return -std::complex<double>(0, 1) * H_eff * state;  // Derivative of the state
    };

    // Compute the RK4 coefficients
    Eigen::VectorXcd k1 = f(psi);
    Eigen::VectorXcd k2 = f(psi + 0.5 * dt * k1);
    Eigen::VectorXcd k3 = f(psi + 0.5 * dt * k2);
    Eigen::VectorXcd k4 = f(psi + dt * k3);

    // Combine the coefficients to get the next state
    return psi + (0.1*dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}
*/
int main(){
// System initialization
Ket psi_0{1,0};
Operator H{{-1, -1},{-1,1}};
Operator l{{1,0},{0,-1}};
double gamma=0.1;
 
std::vector<Eigen::MatrixXcd> L{ std::sqrt(gamma)*l};  // Lindblad operators
Operator H_eff = H;
std::vector<double> jump_probs(L.size());
double total_jump_prob = 0.0;

int dim = psi_0.size();  // Dimension of the Hilbert space

// Initialize total density matrix (set to zero initially)
Eigen::MatrixXcd rho_total = Eigen::MatrixXcd::Zero(dim, dim);
 
int N_traj = 1;  // Number of trajectories
double T = 15;
double dt = 0.15;

// Initialize random number generator with time seed
std::random_device rd;  // Random device for non-deterministic seed
std::mt19937_64 gen(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
//std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
 
// Open a file to write the results
std::ofstream output_file("density_matrix_evolution.dat");
if (!output_file.is_open()) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
}

// Loop over all trajectories
for (int traj = 0; traj < N_traj; ++traj) {
    Eigen::VectorXcd psi = psi_0;  // Initialize psi for each trajectory
    double t = 0.0;

    // Inner loop over time steps
    while (t < T) {
        // Compute jump probabilities and decide if a jump occurs
        std::vector<double> jump_probs(L.size());
        double total_jump_prob = 0.0;
        for (size_t i = 0; i < L.size(); ++i) {
	auto result = (psi.adjoint() * (L[i].adjoint() * L[i]) * psi).value();
	jump_probs[i] = dt * result.real();
            total_jump_prob += jump_probs[i];
        }

        // Generate a random number to decide if a jump occurs
        //double r = static_cast<double>(rand()) / RAND_MAX;  // Random number between 0 and 1
        double r = std::generate_canonical<double, 10>(gen);  // Random number between 0 and 1

        if (r < total_jump_prob) {
            // A jump occurs
            double cumulative_prob = 0.0;
            int jump_operator_index = -1;
            for (size_t i = 0; i < L.size(); ++i) {
                cumulative_prob += jump_probs[i];
                if (r < cumulative_prob) {
                    jump_operator_index = i;
                    break;
                }
            }
            // Apply the selected jump operator
            psi = L[jump_operator_index] * psi;
            //psi =std::sqrt(dt/total_jump_prob)* psi;
            psi.normalize();
            std::cout<<"jump at "<<t<<": "<<r<<std::endl;
        } else {
            // No jump occurs, evolve under non-Hermitian Hamiltonian H_eff
            Eigen::MatrixXcd H_eff = H;
            for (const auto& L_i : L) {
                H_eff -= 0.5i * (L_i.adjoint() * L_i);
            }
            psi = evolve_under_H_eff(H_eff, psi, dt);  // Use Euler method or matrix exp
            //psi =std::sqrt(1./1.-total_jump_prob)* psi;
            psi.normalize();
            std::cout<<"No jump at "<<t<<std::endl;
        }

        // Compute the outer product |psi><psi| for the current time step
        Eigen::MatrixXcd rho = psi * psi.adjoint();  // Density matrix for this time step
	rho = rho/(rho.trace().real());
        // Write time and lower triangular elements to the file
        output_file<< std::fixed << std::setprecision(7)<<std::scientific;
        output_file << t << "\t";
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j <= i; ++j) {  // Lower triangular part
                output_file << std::real(rho(i, j)) << "\t"
                            << std::imag(rho(i, j)) << "\t";
            }
        }
        output_file << std::endl;

        // Advance time
        t += dt;
    }
}

// Close the output file
output_file.close();
std::cout << "Density matrix evolution saved to density_matrix_evolution.dat" << std::endl;

return 0;
}
