#include <Eigen/Core>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
  using namespace Eigen;

  constexpr int m = 4;
  constexpr int n = 3;
  constexpr int nv = 5;

  // Tenseurs 3D : dimensions (m, n, nv), stockage ColMajor
  Tensor<double, 3, ColMajor> H1(m, n, nv);
  Tensor<double, 3, ColMajor> H2(m, n, nv);

  // Remplissage aléatoire
  H1.setRandom();
  H2.setRandom();

  // Vecteur n×1
  VectorXd vec_n = VectorXd::Random(n);

  // Matrices de sortie pour comparaison
  MatrixXd termA_ref(m, nv);
  MatrixXd termA_nocopy(m, nv);

  //
  // --- Méthode "classique" avec chip()
  //
  for (int q = 0; q < nv; ++q) {
    Tensor<double, 2> H1_slice = H1.chip(q, 2);
    Tensor<double, 2> H2_slice = H2.chip(q, 2);

    Map<MatrixXd> M1(const_cast<double *>(H1_slice.data()),
                     H1_slice.dimension(0), H1_slice.dimension(1));
    Map<MatrixXd> M2(const_cast<double *>(H2_slice.data()),
                     H2_slice.dimension(0), H2_slice.dimension(1));

    MatrixXd H_diff = M1 - M2;
    termA_ref.col(q) = H_diff.transpose() * vec_n;
  }

  //
  // --- Méthode optimisée : sans copie / sans malloc
  //
  for (int q = 0; q < nv; ++q) {
    const auto mm = H1.dimension(0);
    const auto nn = H1.dimension(1);

    const double *H1_ptr = H1.data() + q * (mm * nn);
    const double *H2_ptr = H2.data() + q * (mm * nn);

    Map<const Matrix<double, Dynamic, Dynamic, ColMajor>> H1_view(H1_ptr, mm,
                                                                  nn);
    Map<const Matrix<double, Dynamic, Dynamic, ColMajor>> H2_view(H2_ptr, mm,
                                                                  nn);

    termA_nocopy.col(q).noalias() = H1_view.transpose() * vec_n;
    termA_nocopy.col(q).noalias() -= H2_view.transpose() * vec_n;
  }

  //
  // --- Vérification
  //
  double diff = (termA_ref - termA_nocopy).norm();
  std::cout << "Norme de la différence = " << diff << std::endl;

  if (diff < 1e-12)
    std::cout << "✅ Les deux versions donnent les mêmes résultats\n";
  else
    std::cout << "❌ Résultat différent\n";

  return 0;
}