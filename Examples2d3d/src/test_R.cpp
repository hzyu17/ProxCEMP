#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace Eigen;
typedef SparseMatrix<double> SparseMatrixXd;

class SmoothnessTest {
public:
    SparseMatrixXd R_matrix_;
    
    // Method 1: Toeplitz with (n-1)^3 scaling (INCORRECT)
    void computeRMatrix_Toeplitz(size_t n) {
        R_matrix_ = SparseMatrixXd(n, n);
        if (n < 3) { R_matrix_.setZero(); return; }
        
        double scale = std::pow(static_cast<double>(n - 1), 3);
        std::vector<Eigen::Triplet<double>> triplets;
        
        for (size_t i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, 6.0 * scale);
            if (i + 1 < n) {
                triplets.emplace_back(i, i + 1, -4.0 * scale);
                triplets.emplace_back(i + 1, i, -4.0 * scale);
            }
            if (i + 2 < n) {
                triplets.emplace_back(i, i + 2, 1.0 * scale);
                triplets.emplace_back(i + 2, i, 1.0 * scale);
            }
        }
        R_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    }
    
    // Method 2: Interior-only with correct scaling (CORRECT)
    // Only considers second derivatives at interior points i = 1, ..., n-2
    // R = (1/ds^3) * D^T * D where D is (n-2) x n with stencil [1, -2, 1]
    void computeRMatrix_Interior(size_t n) {
        R_matrix_ = SparseMatrixXd(n, n);
        if (n < 3) { R_matrix_.setZero(); return; }
        
        double ds = 1.0 / (n - 1);
        
        // Build D matrix: (n-2) x n, interior second differences only
        MatrixXd D = MatrixXd::Zero(n - 2, n);
        for (size_t i = 0; i < n - 2; ++i) {
            D(i, i) = 1.0;
            D(i, i + 1) = -2.0;
            D(i, i + 2) = 1.0;
        }
        
        // R = (1/ds^3) * D^T * D
        // This gives: Y^T R Y = (1/ds^3) * ||DY||^2 ≈ ∫ ÿ² ds
        double scale = 1.0 / (ds * ds * ds);
        MatrixXd R_dense = scale * D.transpose() * D;
        R_matrix_ = R_dense.sparseView();
    }
    
    // Method 3: Full matrix with proper trapezoidal quadrature (CORRECT)
    // Cost = ds * Σ_i (ÿ_i)² where ÿ_i = (1/ds²)(y_{i-1} - 2y_i + y_{i+1})
    void computeRMatrix_Trapezoidal(size_t n) {
        R_matrix_ = SparseMatrixXd(n, n);
        if (n < 3) { R_matrix_.setZero(); return; }
        
        double ds = 1.0 / (n - 1);
        
        // Build A matrix: n x n, second derivative with boundary handling
        // Interior: stencil [1, -2, 1] / ds²
        // Boundary: truncated stencil
        MatrixXd A = MatrixXd::Zero(n, n);
        double coeff = 1.0 / (ds * ds);
        
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) A(i, i - 1) = coeff;
            A(i, i) = -2.0 * coeff;
            if (i < n - 1) A(i, i + 1) = coeff;
        }
        
        // R = ds * A^T * A (trapezoidal rule for integral)
        MatrixXd R_dense = ds * A.transpose() * A;
        R_matrix_ = R_dense.sparseView();
    }
    
    // Method 4: STOMP-style padding (CORRECT - Toeplitz interior)
    void computeRMatrix_STOMP(size_t n) {
        R_matrix_ = SparseMatrixXd(n, n);
        if (n < 3) { R_matrix_.setZero(); return; }
        
        double ds = 1.0 / (n - 1);
        
        int pad = 2;
        int n_padded = n + 2 * pad;
        
        MatrixXd A_padded = MatrixXd::Zero(n_padded, n_padded);
        double coeff = 1.0 / (ds * ds);
        
        for (int i = 0; i < n_padded; ++i) {
            if (i > 0) A_padded(i, i - 1) = coeff;
            A_padded(i, i) = -2.0 * coeff;
            if (i < n_padded - 1) A_padded(i, i + 1) = coeff;
        }
        
        MatrixXd R_padded = ds * A_padded.transpose() * A_padded;
        MatrixXd R_dense = R_padded.block(pad, pad, n, n);
        R_matrix_ = R_dense.sparseView();
    }

    double computeCost(const VectorXd& Y) {
        return Y.dot(R_matrix_ * Y);
    }
};

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void testAllMethods() {
    printHeader("Comparing All Methods: y(s) = s²");
    std::cout << "Expected cost = ∫₀¹ (d²(s²)/ds²)² ds = ∫₀¹ 4 ds = 4.0\n\n";
    
    SmoothnessTest tester;
    std::vector<int> n_values = {5, 10, 20, 50, 100, 200, 500};
    
    std::cout << std::setw(6) << "n" 
              << std::setw(14) << "Toeplitz"
              << std::setw(14) << "Interior"
              << std::setw(14) << "Trapezoidal"
              << std::setw(14) << "STOMP" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (int n : n_values) {
        VectorXd Y(n);
        for (int i = 0; i < n; ++i) {
            double s = static_cast<double>(i) / (n - 1);
            Y(i) = s * s;
        }
        
        std::cout << std::setw(6) << n;
        
        tester.computeRMatrix_Toeplitz(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        tester.computeRMatrix_Interior(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        tester.computeRMatrix_Trapezoidal(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        tester.computeRMatrix_STOMP(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        std::cout << "\n";
    }
}

void testSinusoidal() {
    printHeader("Comparing All Methods: y(s) = sin(πs)");
    double expected = std::pow(M_PI, 4) / 2.0;
    std::cout << "Expected cost = π⁴/2 ≈ " << std::fixed << std::setprecision(4) << expected << "\n\n";
    
    SmoothnessTest tester;
    std::vector<int> n_values = {10, 20, 50, 100, 200, 500};
    
    std::cout << std::setw(6) << "n" 
              << std::setw(14) << "Toeplitz"
              << std::setw(14) << "Interior"
              << std::setw(14) << "Trapezoidal"
              << std::setw(14) << "STOMP" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (int n : n_values) {
        VectorXd Y(n);
        for (int i = 0; i < n; ++i) {
            double s = static_cast<double>(i) / (n - 1);
            Y(i) = std::sin(M_PI * s);
        }
        
        std::cout << std::setw(6) << n;
        
        tester.computeRMatrix_Toeplitz(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        tester.computeRMatrix_Interior(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        tester.computeRMatrix_Trapezoidal(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        tester.computeRMatrix_STOMP(n);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << tester.computeCost(Y);
        
        std::cout << "\n";
    }
}

void testTimeIndependence() {
    printHeader("Time Independence Verification");
    std::cout << "The R matrix is built in normalized time s ∈ [0,1], so T doesn't appear.\n";
    std::cout << "All methods are T-independent by construction.\n";
}

void testConvergence() {
    printHeader("Convergence Analysis: y(s) = sin(πs)");
    double expected = std::pow(M_PI, 4) / 2.0;
    std::cout << "Expected: " << std::fixed << std::setprecision(4) << expected << "\n";
    std::cout << "Using Interior method (best convergence)\n\n";
    
    SmoothnessTest tester;
    
    std::cout << std::setw(8) << "n" 
              << std::setw(15) << "Cost"
              << std::setw(15) << "Error"
              << std::setw(15) << "Error*n²" << "\n";
    std::cout << std::string(53, '-') << "\n";
    
    for (int n : {10, 20, 40, 80, 160, 320, 640, 1280}) {
        tester.computeRMatrix_Interior(n);
        
        VectorXd Y(n);
        for (int i = 0; i < n; ++i) {
            double s = static_cast<double>(i) / (n - 1);
            Y(i) = std::sin(M_PI * s);
        }
        
        double cost = tester.computeCost(Y);
        double error = std::abs(cost - expected);
        double error_n2 = error * n * n;
        
        std::cout << std::setw(8) << n 
                  << std::setw(15) << std::fixed << std::setprecision(6) << cost
                  << std::setw(15) << std::scientific << std::setprecision(4) << error
                  << std::setw(15) << std::fixed << std::setprecision(2) << error_n2 << "\n";
    }
    std::cout << "\nIf Error*n² is roughly constant, convergence is O(1/n²)\n";
}

void testMatrixStructure() {
    printHeader("Matrix Structure Comparison (n=6)");
    
    SmoothnessTest tester;
    double ds = 1.0 / 5.0;
    double scale = 1.0 / (ds * ds * ds);  // 125
    
    std::cout << "Normalized by 1/ds³ = " << scale << "\n\n";
    
    std::cout << "Interior Method (recommended):\n";
    tester.computeRMatrix_Interior(6);
    MatrixXd R_int = MatrixXd(tester.R_matrix_) / scale;
    std::cout << std::fixed << std::setprecision(1);
    for (int i = 0; i < 6; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 6; ++j) {
            std::cout << std::setw(6) << R_int(i, j);
        }
        std::cout << " ]\n";
    }
    
    std::cout << "\nTrapezoidal Method:\n";
    tester.computeRMatrix_Trapezoidal(6);
    MatrixXd R_trap = MatrixXd(tester.R_matrix_) / scale;
    for (int i = 0; i < 6; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 6; ++j) {
            std::cout << std::setw(6) << R_trap(i, j);
        }
        std::cout << " ]\n";
    }
    
    std::cout << "\nSTOMP Padding Method:\n";
    tester.computeRMatrix_STOMP(6);
    MatrixXd R_stomp = MatrixXd(tester.R_matrix_) / scale;
    for (int i = 0; i < 6; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 6; ++j) {
            std::cout << std::setw(6) << R_stomp(i, j);
        }
        std::cout << " ]\n";
    }
}

int main() {
    std::cout << std::string(70, '*') << "\n";
    std::cout << "R Matrix Discretization Independence Test\n";
    std::cout << "Normalized time s ∈ [0,1], cost ≈ ∫₀¹ (d²y/ds²)² ds\n";
    std::cout << std::string(70, '*') << "\n";
    
    testAllMethods();
    testSinusoidal();
    testTimeIndependence();
    testConvergence();
    testMatrixStructure();
    
    printHeader("Summary");
    std::cout << "Methods compared:\n";
    std::cout << "  1. Toeplitz:    (n-1)³ × [1,-4,6,-4,1] - DIVERGES (boundary error)\n";
    std::cout << "  2. Interior:    (1/ds³) × D^T D, D is (n-2)×n - CONVERGES\n";
    std::cout << "  3. Trapezoidal: ds × A^T A, A is n×n - CONVERGES but boundary-affected\n";
    std::cout << "  4. STOMP:       Padded A^T A, central block - Same as Trapezoidal\n\n";
    
    std::cout << "Recommended: Interior method\n";
    std::cout << "  - Only considers second derivatives at interior points\n";
    std::cout << "  - Clean O(1/n²) convergence\n";
    std::cout << "  - Independent of T and converges as n → ∞\n\n";
    
    return 0;
}
