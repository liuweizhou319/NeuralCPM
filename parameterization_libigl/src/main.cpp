#include <cassert>

#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/frame_field_deformer.h>
#include <igl/frame_to_cross_field.h>
#include <igl/local_basis.h>
#include <igl/readOBJ.h>
#include <igl/rotate_vectors.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/copyleft/comiso/miq.h>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <Eigen/Dense>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using MatrixX2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;

std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::filesystem::path resolve_input_path(const std::filesystem::path& arg_path, const char* argv0) {
  if (arg_path.is_absolute() && std::filesystem::exists(arg_path)) {
    return arg_path;
  }

  if (std::filesystem::exists(arg_path)) {
    return std::filesystem::absolute(arg_path);
  }

  std::filesystem::path base = std::filesystem::absolute(argv0).parent_path();
  for (int i = 0; i < 8; ++i) {
    const auto candidate = base / arg_path;
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
    if (!base.has_parent_path()) {
      break;
    }
    base = base.parent_path();
  }

  return arg_path;
}

void load_face_frame_field(
    const std::filesystem::path& path,
    int face_count,
    Eigen::MatrixXd& ff1,
    Eigen::MatrixXd& ff2) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open frame field file: " + path.string());
  }

  std::vector<std::array<double, 6>> rows;
  std::string line;
  int line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    const std::string cleaned = trim(line);
    if (cleaned.empty() || cleaned[0] == '#') {
      continue;
    }

    std::istringstream iss(cleaned);
    std::array<double, 6> values{};
    for (int i = 0; i < 6; ++i) {
      if (!(iss >> values[i])) {
        throw std::runtime_error(
            "invalid frame field line " + std::to_string(line_number) +
            ", expected 6 floating-point values");
      }
    }

    double extra = 0.0;
    if (iss >> extra) {
      throw std::runtime_error(
          "invalid frame field line " + std::to_string(line_number) +
          ", got more than 6 values");
    }
    rows.push_back(values);
  }

  if (static_cast<int>(rows.size()) != face_count) {
    throw std::runtime_error(
        "frame field row count mismatch: expected " + std::to_string(face_count) +
        ", got " + std::to_string(rows.size()));
  }

  ff1.resize(face_count, 3);
  ff2.resize(face_count, 3);
  for (int i = 0; i < face_count; ++i) {
    for (int j = 0; j < 3; ++j) {
      ff1(i, j) = rows[i][j];
      ff2(i, j) = rows[i][j + 3];
    }
  }
}

MatrixX2d build_corner_uv(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& f_uv) {
  MatrixX2d corner_uv(f_uv.rows() * 3, 2);
  for (int i = 0; i < f_uv.rows(); ++i) {
    for (int c = 0; c < 3; ++c) {
      corner_uv.row(i * 3 + c) = uv.row(f_uv(i, c));
    }
  }
  return corner_uv;
}

std::filesystem::path default_output_path(const std::filesystem::path& mesh_path) {
  return mesh_path.parent_path() / (mesh_path.stem().string() + "_uv.obj");
}

void write_obj_with_uv(
    const std::filesystem::path& output_path,
    const Eigen::MatrixXd& v,
    const Eigen::MatrixXi& f,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& f_uv) {
  if (f.rows() != f_uv.rows() || f.cols() != 3 || f_uv.cols() != 3) {
    throw std::runtime_error("invalid face topology for OBJ export");
  }

  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("failed to open output OBJ: " + output_path.string());
  }

  out << std::fixed << std::setprecision(17);
  out << "# surfaceframefield_igl uv export\n";

  for (int i = 0; i < v.rows(); ++i) {
    out << "v " << v(i, 0) << ' ' << v(i, 1) << ' ' << v(i, 2) << '\n';
  }

  for (int i = 0; i < uv.rows(); ++i) {
    out << "vt " << uv(i, 0) << ' ' << uv(i, 1) << '\n';
  }

  for (int i = 0; i < f.rows(); ++i) {
    out << "f ";
    for (int c = 0; c < 3; ++c) {
      out << (f(i, c) + 1) << '/' << (f_uv(i, c) + 1);
      if (c < 2) {
        out << ' ';
      }
    }
    out << '\n';
  }
}

void visualize_result(
    const Eigen::MatrixXd& v,
    const Eigen::MatrixXi& f,
    const Eigen::MatrixXd& ff1,
    const Eigen::MatrixXd& ff2,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& f_uv) {
  const double edge_scale = igl::avg_edge_length(v, f);

  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  auto* input_mesh = polyscope::registerSurfaceMesh("input mesh", v, f);
  input_mesh->setEdgeWidth(1.0);
  input_mesh->addFaceVectorQuantity("frame x", ff1)
      ->setVectorColor(glm::vec3(0.90f, 0.20f, 0.15f))
      ->setVectorLengthScale(0.75 * edge_scale, false)
      ->setEnabled(true);
  input_mesh->addFaceVectorQuantity("frame y", ff2)
      ->setVectorColor(glm::vec3(0.10f, 0.35f, 0.90f))
      ->setVectorLengthScale(0.75 * edge_scale, false)
      ->setEnabled(true);

  const MatrixX2d corner_uv = build_corner_uv(uv, f_uv);
  input_mesh->addParameterizationQuantity("seamless uv", corner_uv, polyscope::ParamCoordsType::WORLD)
      ->setEnabled(true);

  // auto* uv_mesh = polyscope::registerSurfaceMesh2D("uv domain", uv, f_uv);
  // uv_mesh->setEdgeWidth(1.0);

  polyscope::show();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " <mesh.obj> <face_frame_field.txt> [output_uv.obj]\n";
      return 1;
    }

    const std::filesystem::path mesh_path = resolve_input_path(argv[1], argv[0]);
    const std::filesystem::path field_path = resolve_input_path(argv[2], argv[0]);
    const std::filesystem::path output_path =
        argc >= 4 ? std::filesystem::absolute(argv[3]) : default_output_path(mesh_path);

    Eigen::MatrixXd v;
    Eigen::MatrixXi f;
    if (!igl::readOBJ(mesh_path.string(), v, f)) {
      throw std::runtime_error("failed to read OBJ mesh: " + mesh_path.string());
    }
    if (f.cols() != 3) {
      throw std::runtime_error("the input OBJ must be a triangular mesh");
    }

    Eigen::MatrixXd ff1;
    Eigen::MatrixXd ff2;
    load_face_frame_field(field_path, f.rows(), ff1, ff2);

    std::cout << "deforming frame field...\n";
    Eigen::MatrixXd v_deformed;
    Eigen::MatrixXd ff1_deformed;
    Eigen::MatrixXd ff2_deformed;
    igl::frame_field_deformer(v, f, ff1, ff2, v_deformed, ff1_deformed, ff2_deformed);

    std::cout << "converting to cross field...\n";
    Eigen::MatrixXd x1_deformed;
    igl::frame_to_cross_field(v_deformed, f, ff1_deformed, ff2_deformed, x1_deformed);

    Eigen::MatrixXd b1;
    Eigen::MatrixXd b2;
    Eigen::MatrixXd b3;
    igl::local_basis(v_deformed, f, b1, b2, b3);
    Eigen::MatrixXd x2_deformed = igl::rotate_vectors(
        x1_deformed,
        Eigen::VectorXd::Constant(1, igl::PI / 2.0),
        b1,
        b2);

    Eigen::MatrixXd uv;
    Eigen::MatrixXi f_uv;
    Eigen::MatrixXi tt;
    igl::triangle_triangle_adjacency(f, tt);
    std::vector<std::vector<int>> hard_features;
    for (int fi = 0; fi < f.rows(); ++fi) {
      for (int e = 0; e < 3; ++e) {
        if (tt(fi, e) == -1) {
          hard_features.push_back({fi, e});
        }
      }
    }

    std::cout << "running miq...\n";
    igl::copyleft::comiso::miq(
        v_deformed,
        f,
        x1_deformed,
        x2_deformed,
        uv,
        f_uv,
        60.0, // 60.0
        5.0,
        false,
        5, // stiffening iterations，原来是 2
        5, // integer rounding local iterations
        true,
        true,
        std::vector<int>{},
        hard_features);
    std::cout << "miq finished.\n";

    std::cout << "mesh vertices: " << v.rows() << "\n";
    std::cout << "mesh faces: " << f.rows() << "\n";
    std::cout << "uv vertices: " << uv.rows() << "\n";
    write_obj_with_uv(output_path, v, f, uv, f_uv);
    std::cout << "wrote obj with uv: " << output_path.string() << "\n";

    visualize_result(v, f, ff1, ff2, uv, f_uv);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
