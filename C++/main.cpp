#include <iostream>
#include <vector>
#include <chrono>
#include <fftw3.h>
#include <sndfile.h>
#include <fstream>

// Function to perform overlap-add convolution
std::vector<double> overlap_add_convolution(const std::vector<double>& signal, const std::vector<double>& filter, int block_size) {
    int signal_length = signal.size();
    int filter_length = filter.size();
    int output_length = signal_length + filter_length - 1;

    // FFT size must be at least (block_size + filter_length - 1) to avoid circular convolution artifacts
    int fft_size = 1;
    while (fft_size < block_size + filter_length - 1) {
        fft_size *= 2;
    }

    // Allocate memory for FFTW
    std::vector<double> padded_signal(fft_size, 0.0);
    std::vector<double> padded_filter(fft_size, 0.0);
    fftw_complex *fft_signal = fftw_alloc_complex(fft_size / 2 + 1);
    fftw_complex *fft_filter = fftw_alloc_complex(fft_size / 2 + 1);
    fftw_complex *fft_result = fftw_alloc_complex(fft_size / 2 + 1);

    // Create FFTW plans
    fftw_plan plan_forward_signal = fftw_plan_dft_r2c_1d(fft_size, padded_signal.data(), fft_signal, FFTW_ESTIMATE);
    fftw_plan plan_forward_filter = fftw_plan_dft_r2c_1d(fft_size, padded_filter.data(), fft_filter, FFTW_ESTIMATE);
    fftw_plan plan_inverse = fftw_plan_dft_c2r_1d(fft_size, fft_result, padded_signal.data(), FFTW_ESTIMATE);

    // Perform FFT on the filter (only once)
    std::copy(filter.begin(), filter.end(), padded_filter.begin());
    fftw_execute(plan_forward_filter);

    // Output vector
    std::vector<double> output(output_length, 0.0);

    // Process signal in blocks
    for (int i = 0; i < signal_length; i += block_size) {
        int block_start = i;
        int block_end = std::min(i + block_size, signal_length);
        int block_length = block_end - block_start;

        // Copy the current block to the padded signal vector
        std::fill(padded_signal.begin(), padded_signal.end(), 0.0);
        std::copy(signal.begin() + block_start, signal.begin() + block_end, padded_signal.begin());

        // Perform FFT on the block
        fftw_execute(plan_forward_signal);

        // Multiply the FFT of the block with the FFT of the filter
        for (int k = 0; k < fft_size / 2 + 1; k++) {
            fft_result[k][0] = fft_signal[k][0] * fft_filter[k][0] - fft_signal[k][1] * fft_filter[k][1];
            fft_result[k][1] = fft_signal[k][0] * fft_filter[k][1] + fft_signal[k][1] * fft_filter[k][0];
        }

        // Perform inverse FFT to get the time-domain result
        fftw_execute(plan_inverse);

        // Overlap and add the result to the output
        for (int j = 0; j < fft_size; j++) {
            if (block_start + j < output_length) {
                output[block_start + j] += padded_signal[j] / fft_size;
            }
        }
    }

    // Clean up FFTW
    fftw_destroy_plan(plan_forward_signal);
    fftw_destroy_plan(plan_forward_filter);
    fftw_destroy_plan(plan_inverse);
    fftw_free(fft_signal);
    fftw_free(fft_filter);
    fftw_free(fft_result);

    return output;
}

int main() {
    // File names
    const char *input_filename = "input.wav";
    const char *ir_filename = "filtr1024.wav";
    const char *output_filename = "output.wav";

    // Read the input WAV file using libsndfile
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(input_filename, SFM_READ, &sfinfo);
    if (!infile) {
        std::cerr << "Error opening file: " << input_filename << std::endl;
        return 1;
    }

    // Assume the file is mono (1 channel)
    if (sfinfo.channels != 1) {
        std::cerr << "File must be mono (1 channel)!" << std::endl;
        sf_close(infile);
        return 1;
    }

    // Read the signal
    long num_samples = sfinfo.frames;
    std::vector<double> signal(num_samples);
    if (sf_read_double(infile, signal.data(), num_samples) != num_samples) {
        std::cerr << "Error reading samples from file." << std::endl;
        sf_close(infile);
        return 1;
    }
    sf_close(infile);
    std::cout << "Read " << num_samples << " samples." << std::endl;

    SNDFILE *infileir = sf_open(ir_filename, SFM_READ, &sfinfo);
    if (!infileir) {
        std::cerr << "Error opening file: " << ir_filename << std::endl;
        return 1;
    }

    // Assume the file is mono (1 channel)
    if (sfinfo.channels != 1) {
        std::cerr << "File must be mono (1 channel)!" << std::endl;
        sf_close(infileir);
        return 1;
    }

    // Read the signal
    long ir_samples = sfinfo.frames;
    std::vector<double> filter(ir_samples);
    if (sf_read_double(infileir, filter.data(), ir_samples) != ir_samples) {
        std::cerr << "Error reading samples from file." << std::endl;
        sf_close(infileir);
        return 1;
    }
    sf_close(infileir);
    std::cout << "Read " << ir_samples << " samples." << std::endl;


    // Perform overlap-add convolution
    // Time measurements for adaptive block length
    std::string str;
    std::vector<double> output;
    auto startever = std::chrono::high_resolution_clock::now();

    for(int q = 0; q<50; q++) {
        int block_size = ir_samples;
        // int block_size = 1024;

        auto start = std::chrono::high_resolution_clock::now();
        output = overlap_add_convolution(signal, filter, block_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Computation time: " << elapsed.count() << " s" << std::endl;
        str.append(std::to_string(elapsed.count()) + ",\n");
        if (std::chrono::high_resolution_clock::now() - startever > std::chrono::seconds(60)) {
            break;
        }
    }

    std::ofstream out("timesadaptive" + std::to_string(ir_samples) +".txt");
    out << str;
    out.close();


    // Time measurements for 1024 block length

    std::string strf;
    std::vector<double> outputc;
    for(int q = 0; q<50; q++) {
        // int block_size = ir_samples;
        int block_size = 1024;

        auto start = std::chrono::high_resolution_clock::now();
        outputc = overlap_add_convolution(signal, filter, block_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Computation time: " << elapsed.count() << " s" << std::endl;
        strf.append(std::to_string(elapsed.count()) + ",\n");
        if (std::chrono::high_resolution_clock::now() - startever > std::chrono::seconds(60)) {
            break;
        }
    }

    std::ofstream outf("timesfixed" + std::to_string(ir_samples) +".txt");
    outf << strf;
    outf.close();





    // Truncate the output to match the input length
    output.resize(num_samples);

    // Write the output to a new WAV file
    SF_INFO out_sfinfo;
    out_sfinfo.frames = output.size();
    out_sfinfo.samplerate = sfinfo.samplerate;
    out_sfinfo.channels = 1;
    out_sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE *outfile = sf_open(output_filename, SFM_WRITE, &out_sfinfo);
    if (!outfile) {
        std::cerr << "Error creating file: " << output_filename << std::endl;
        return 1;
    }
    sf_count_t num_written = sf_write_double(outfile, output.data(), output.size());
    if (num_written != output.size()) {
        std::cerr << "Error writing samples to file." << std::endl;
    }
    sf_close(outfile);
    std::cout << "File saved as " << output_filename << std::endl;

    return 0;
}