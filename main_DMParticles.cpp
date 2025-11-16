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
    double x, y, z, vx, vy, vz;
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

        selected_particles.push_back(p);
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
    long int total_particles = total_floats / 6;

    // Divide particles among ranks
    long int base_count = total_particles / size;
    long int remainder = total_particles % size;
    long int local_count = (rank < remainder) ? base_count + 1 : base_count;
    long int offset = rank * base_count + std::min(rank, (int)remainder);

    MPI_Offset byte_offset = header_size + offset * 6 * sizeof(float);

    // Read local data
    std::vector<float> data(6 * local_count);
    MPI_File_read_at_all(mpi_file, byte_offset, data.data(), data.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    // Convert to LightConeParticle
    std::vector<LightConeParticle> local_particles(local_count);
    for (long int i = 0; i < local_count; ++i) {
        float x = data[6*i], y = data[6*i+1], z = data[6*i+2];
        float vx = data[6*i+3], vy = data[6*i+4], vz = data[6*i+5];

        SwapEnd(x); SwapEnd(y); SwapEnd(z);
        SwapEnd(vx); SwapEnd(vy); SwapEnd(vz);

        local_particles[i] = {x, y, z, vx, vy, vz};
    }

    // Extract particles in solid angle
    extractParticlesInSolidAngle(local_particles, solid_angle_particles, xcen, ycen, zcen, solid_angle_rad);
}

void writeBinaryVTK(const std::string& filename, const std::vector<LightConeParticle>& particles) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long int local_num_particles = particles.size();
    long int total_num_particles = 0;

    // Get total particles across all ranks
    MPI_Allreduce(&local_num_particles, &total_num_particles, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Compute offset for this rank
    size_t offset = 0;
    MPI_Exscan(&local_num_particles, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Header handling
    size_t header_size = 0;
    if (rank == 0) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Could not open file " << filename << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Write the header
        file << "# vtk DataFile Version 2.0\n";
        file << "Particle Cloud Data\n";
        file << "BINARY\n";
        file << "DATASET POLYDATA\n";
        file << "POINTS " << total_num_particles << " float\n";

        // Determine header size
        file.seekp(0, std::ios::end);
        header_size = file.tellp();
        file.close();
    }

    // Broadcast header size to all ranks
    MPI_Bcast(&header_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Prepare local particle data (x, y, z only)
    std::vector<float> data(3 * local_num_particles);
    for (size_t i = 0; i < local_num_particles; ++i) {
        data[3 * i]     = static_cast<float>(particles[i].x);
        data[3 * i + 1] = static_cast<float>(particles[i].y);
        data[3 * i + 2] = static_cast<float>(particles[i].z);

        SwapEnd(data[3 * i]);
        SwapEnd(data[3 * i + 1]);
        SwapEnd(data[3 * i + 2]);
    }

    // MPI collective I/O
    MPI_File mpi_file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, &mpi_file);

    MPI_Offset byte_offset = static_cast<MPI_Offset>(header_size + 3 * offset * sizeof(float));
    MPI_File_write_at_all(mpi_file, byte_offset, data.data(), data.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    if (rank == 0) {
        std::cout << "Successfully wrote VTK file: " << filename << "\n";
    }
}




int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string directory = "/lustre/orion/scratch/nataraj2/geo163/HaloFinder_GPU/Output_8192/LightCones/SimpleBinary";

    double xcen = 3850.0;
    double ycen = 3850.0;
    double zcen = 3850.0;
    double solid_angle_rad = 2.0*0.5236/5.0;

    std::vector<LightConeParticle> rank_selected_particles;

    // Get sorted list of all *.bin files
    std::vector<fs::path> bin_files;
    for (auto& p : fs::directory_iterator(directory)) {
        if (p.path().extension() == ".bin") bin_files.push_back(p.path());
    }
    std::sort(bin_files.begin(), bin_files.end());

    // Loop over files and accumulate selected particles
    for (size_t i = 7; i <= 7; i += 2) {
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

        // End timer
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        if (rank == 0) {
            std::cout << "Time taken to process file: " << file_path 
                  << " = " << elapsed << " seconds.\n";
        }
        
    }

    //std::cout << "Rank " << rank << " has accumulated " << rank_selected_particles.size()
      //        << " particles in solid angle.\n";

    writeBinaryVTK("LightConeSolidAngle.vtk",rank_selected_particles);

    MPI_Finalize();
    return 0;
}
