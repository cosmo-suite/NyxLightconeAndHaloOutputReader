#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;

struct LightConeParticle {
    double x, y, z, mass;
    int ncells;
};

void SwapEnd(float& val) {
    char* bytes = reinterpret_cast<char*>(&val);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);
}


void extractParticlesInSolidAngle(const std::vector<LightConeParticle>& all_particles,
                                  std::vector<LightConeParticle>& selected_particles,
                                  double xcen, double ycen, double zcen,
                                  double solid_angle_rad) {
    for (const auto& p : all_particles) {
        double dx = p.x - xcen;
        double dy = p.y - ycen;
        double dz = p.z - zcen;
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-12) continue;
        double theta = std::acos(dz / r);
        //if (theta < solid_angle_rad) selected_particles.push_back(p);

        //if(p.mass > 1e13){
        selected_particles.push_back(p);
        //}
    }
}

void readBinaryAndExtractSolidAngle(const std::string& filename,
                                    std::vector<LightConeParticle>& solid_angle_particles,
                                    double xcen, double ycen, double zcen,
                                    double solid_angle_rad) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open MPI file
    MPI_File mpi_file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

    // Get file size
    MPI_Offset file_size;
    MPI_File_get_size(mpi_file, &file_size);

    MPI_Offset header_size = 0; // adjust if header exists
    long int total_floats = (file_size - header_size) / sizeof(float);
    long int total_particles = total_floats / 5;

    // Divide particles among ranks
    long int base_count = total_particles / size;
    long int remainder = total_particles % size;
    long int local_count = (rank < remainder) ? base_count + 1 : base_count;
    long int offset = rank * base_count + std::min(rank, (int)remainder);

    MPI_Offset byte_offset = header_size + offset * 5 * sizeof(float);

    // Read local data
    std::vector<float> data(5 * local_count);
    MPI_File_read_at_all(mpi_file, byte_offset, data.data(), data.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    // Convert to LightConeParticle
    std::vector<LightConeParticle> local_particles(local_count);
    for (long int i = 0; i < local_count; ++i) {
        float x = data[5*i], y = data[5*i+1], z = data[5*i+2];
        float mass = data[5*i+3], ncells = data[5*i+4];

        SwapEnd(x); SwapEnd(y); SwapEnd(z);
        SwapEnd(mass); SwapEnd(ncells);

        local_particles[i].x = x;
        local_particles[i].y = y;
        local_particles[i].z = z;
        local_particles[i].mass = mass;
        local_particles[i].ncells = static_cast<int>(ncells);
    }

    // Extract particles in solid angle
    extractParticlesInSolidAngle(local_particles, solid_angle_particles, xcen, ycen, zcen, solid_angle_rad);
}

void writeBinaryVTK(const std::string& filename,
                    const std::vector<LightConeParticle>& particles) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long local_num = particles.size();
    long total_num = 0;

    MPI_Allreduce(&local_num, &total_num, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Compute offsets
    MPI_Offset offset = 0;
    MPI_Exscan(&local_num, &offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    // --- Rank 0 writes ASCII headers ---
    size_t points_header_size = 0;
    size_t scalar_header_size = 0;

    if (rank == 0) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) MPI_Abort(MPI_COMM_WORLD, 1);

        // VTK header
        file << "# vtk DataFile Version 2.0\n";
        file << "Particle Cloud Data\n";
        file << "BINARY\n";
        file << "DATASET POLYDATA\n";
        file << "POINTS " << total_num << " float\n";

        file.seekp(0, std::ios::end);
        points_header_size = file.tellp();
        file.close();
    }

    MPI_Bcast(&points_header_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // --- Write points binary ---
    std::vector<float> points_data(3 * local_num);
    for (long i = 0; i < local_num; ++i) {
        points_data[3*i]   = static_cast<float>(particles[i].x);
        points_data[3*i+1] = static_cast<float>(particles[i].y);
        points_data[3*i+2] = static_cast<float>(particles[i].z);
        SwapEnd(points_data[3*i]);
        SwapEnd(points_data[3*i+1]);
        SwapEnd(points_data[3*i+2]);
    }

    MPI_File mpi_file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);
    MPI_Offset points_offset = static_cast<MPI_Offset>(points_header_size + 3*offset*sizeof(float));
    MPI_File_write_at_all(mpi_file, points_offset, points_data.data(), points_data.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    // --- Rank 0 writes scalar ASCII header ---
    if (rank == 0) {
        std::ofstream file(filename, std::ios::binary | std::ios::app);
        if (!file) MPI_Abort(MPI_COMM_WORLD, 1);

        file << "POINT_DATA " << total_num << "\n";
        file << "SCALARS mass float 1\n";
        file << "LOOKUP_TABLE default\n";

        file.seekp(0, std::ios::end);
        scalar_header_size = file.tellp();
        file.close();
    }

    MPI_Bcast(&scalar_header_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // --- Write scalar binary ---
    std::vector<float> scalar_data(local_num);
    for (long i = 0; i < local_num; ++i) {
        scalar_data[i] = static_cast<float>(particles[i].mass);
        SwapEnd(scalar_data[i]);
    }

    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);
    MPI_Offset scalar_offset = static_cast<MPI_Offset>(scalar_header_size + offset*sizeof(float));
    MPI_File_write_at_all(mpi_file, scalar_offset, scalar_data.data(), scalar_data.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    if (rank == 0) {
        std::cout << "Successfully wrote VTK file: " << filename << "\n";
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string directory = "/lustre/orion/scratch/nataraj2/geo163/HaloFinder_GPU/Output_8192/Halos/SimpleBinary";

    double xcen = 3850.0;
    double ycen = 3850.0;
    double zcen = 3850.0;
    double solid_angle_rad = 0.5236;

    std::vector<LightConeParticle> rank_selected_particles;

    // Get sorted list of all *.bin files
    std::vector<fs::path> bin_files;
    for (auto& p : fs::directory_iterator(directory)) {
        if (p.path().extension() == ".bin") bin_files.push_back(p.path());
    }
    std::sort(bin_files.begin(), bin_files.end());

    long int total_num_halos = 0;

    // After MPI is initialized
    std::ofstream ofs;
    if (rank == 0) {
        ofs.open("halos_vs_redshift.txt");
        ofs << "# redshift   num_halos\n";   // header
    }

    // Loop over files and accumulate selected particles
    for (size_t i = 0; i < bin_files.size(); i += 1) {
        const auto& file_path = bin_files[i];

        // Start timer
        double t_start = MPI_Wtime();
        if (rank == 0) std::cout << "Processing file: " << file_path << std::endl;

        std::vector<LightConeParticle> file_selected_particles;
        readBinaryAndExtractSolidAngle(file_path.string(), file_selected_particles,
                                       xcen, ycen, zcen, solid_angle_rad);

        // Append to the rank's accumulated vector
        rank_selected_particles.insert(rank_selected_particles.end(),
                                      file_selected_particles.begin(),
                                      file_selected_particles.end());

        // --- MPI reduction to get total number of particles for this file ---
        long long local_count  = static_cast<long long>(file_selected_particles.size());
        long long global_count = 0;
        MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {

            // Extract redshift from filename: e.g. "reeber_halos_0000395.bin"
            std::string fname = file_path.filename().string();
            size_t uscore = fname.find_last_of('_');
            size_t dot    = fname.find_last_of('.');
            std::string num_str = fname.substr(uscore+1, dot-uscore-1); // "0000395"

            double redshift = std::stod(num_str) / 100.0;  // 0000395 → 3.95

            // Append to text file
            ofs << std::fixed << std::setprecision(2)
                << redshift << " "
                << global_count << "\n";

            std::cout << "Redshift " << redshift
                      << " → total halos = " << global_count << std::endl;
        
            std::cout << "Total number of halos in file " << file_path
                  << " = " << global_count << std::endl;
            total_num_halos += global_count;
        }

        // End timer
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        if (rank == 0) {
            std::cout << "Time taken to process file: " << file_path 
                  << " = " << elapsed << " seconds.\n";
        }
        
    }

    if (rank == 0) {
        std::cout << "The total number of halos is " << total_num_halos << std::endl;
    }

    //std::cout << "Rank " << rank << " has accumulated " << rank_selected_particles.size()
      //        << " particles in solid angle.\n";

    writeBinaryVTK("LightConeSolidAngle_Halos.vtk",rank_selected_particles);

    MPI_Finalize();
    return 0;
}
