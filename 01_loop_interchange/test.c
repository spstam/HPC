#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

// --- Configuration Constants ---
#define PROGRAM_PATH "./sobelv1"
#define REFERENCE_FILE "prototype.grey"
#define OUTPUT_FILE "output_sobel.grey"
#define CSV_OUTPUT_FILE "benchmark_times.csv" // CSV file for results
#define N_RUNS 21
// Image dimensions based on 16,777,216 bytes (4096 * 4096)
#define IMAGE_SIZE (4096 * 4096)
// -------------------------------

// Comparator function for qsort
int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;

    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

/**
 * @brief Executes the Sobel program and extracts the execution time.
 * @param program_path Path to the executable.
 * @return double The execution time in seconds, or -1.0 on failure.
 */
double run_and_time(const char *program_path) {
    FILE *fp;
    char buffer[256];
    double time_sec = -1.0;
    double PSNR;
    // Use popen to execute the program and capture output
    fp = popen(program_path, "r");
    if (fp == NULL) {
        perror("Error executing program with popen");
        return -1.0;
    }

    // Read output line by line
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Look for the timing pattern: "Total time =   1.99207 seconds"
        // sscanf is used to parse the time value
        if (sscanf(buffer, "Total time = %lf seconds", &time_sec) == 1) {
            // Found the time, no need to read further
            continue;
        }
        if(sscanf(buffer,"PSNR of original Sobel and computed Sobel image: %lg\n", &PSNR) == 1){
            printf("PSNR= %lg\n", PSNR);
            continue;
        }
    }

    if (pclose(fp) == -1) {
        perror("Error closing program stream (pclose)");
        // We still return the time if it was found
    }

    // Return the parsed time or -1.0 if not found/error
    return time_sec;
}

/**
 * @brief Compares two binary files byte-by-byte for exact equality.
 * @param file1_path Path to the first file (e.g., output_sobel.grey).
 * @param file2_path Path to the second file (e.g., prototype.grey).
 * @return int 0 if identical, 1 if different, -1 on I/O error.
 */
int compare_files(const char *file1_path, const char *file2_path) {
    FILE *fp1, *fp2;
    int identical = 0; // Assume identical until proven otherwise
    
    // Check file sizes first (more efficient than byte-by-byte)
    struct stat st1, st2;
    if (stat(file1_path, &st1) != 0 || stat(file2_path, &st2) != 0) {
        perror("Error checking file size (stat)");
        return -1;
    }

    if (st1.st_size != st2.st_size || st1.st_size != IMAGE_SIZE) {
        printf("Error: Files have mismatched sizes or size does not match expected IMAGE_SIZE (%zu vs %d).\n", (size_t)st1.st_size, IMAGE_SIZE);
        return 1; // Different size is a significant difference
    }

    // Open files in binary read mode
    fp1 = fopen(file1_path, "rb");
    fp2 = fopen(file2_path, "rb");

    if (fp1 == NULL || fp2 == NULL) {
        perror("Error opening .grey files");
        if (fp1) fclose(fp1);
        if (fp2) fclose(fp2);
        return -1;
    }

    // Compare byte-by-byte
    unsigned char byte1, byte2;
    size_t count = 0;
    while (count < IMAGE_SIZE) {
        if (fread(&byte1, 1, 1, fp1) != 1 || fread(&byte2, 1, 1, fp2) != 1) {
            fprintf(stderr, "Error: Unexpected end of file or read error.\n");
            identical = -1;
            break;
        }

        if (byte1 != byte2) {
            identical = 1; // Files are different
            break; 
        }
        count++;
    }

    fclose(fp1);
    fclose(fp2);
    
    return identical;
}


int main() {
    double times[N_RUNS];
    double current_time;
    double sum = 0.0;
    double average_time;
    double sum_sq_diff = 0.0;
    double std_dev = 0.0;
    int i;
    int comparison_result;

    printf("--- HPC Benchmarking and Verification (%d Runs) ---\n", N_RUNS);

    // 1. Benchmark Execution
    for (i = 0; i < N_RUNS; i++) {
        // --- ADD THIS BLOCK TO CLEAR THE CACHE ---
        printf("Clearing system page cache... (Run %d/%d)\n", i + 1, N_RUNS);
        // The command requires sudo. We redirect output to /dev/null to keep the console clean.
        int ret = system("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null");
        if (ret != 0) {
            fprintf(stderr, "Warning: Failed to clear cache. Did you run with 'sudo'?\n");
            // You might want to exit here if a clean cache is essential
        }
        // ------------------------------------------

        printf("Running %s...\n", PROGRAM_PATH);
        current_time = run_and_time(PROGRAM_PATH);

        if (current_time < 0) {
            fprintf(stderr, "Fatal: Run %d failed. Stopping benchmark.\n", i + 1);
            return 1;
        }
        times[i] = current_time;
        printf("%.6f\n", times[i]);
    }

    // 2. Trimmed Average and Standard Deviation Calculation
    
    // Sort the array of times
    qsort(times, N_RUNS, sizeof(double), compare_doubles);
    
    // Check for minimum runs requirement (N >= 3)
    if (N_RUNS < 3) {
        fprintf(stderr, "Error: Cannot perform statistics with N < 3.\n");
        return 1;
    }
    
    // Sum times, excluding the lowest (index 0) and highest (index N_RUNS - 1)
    for (i = 1; i < N_RUNS - 1; i++) {
        sum += times[i];
    }
    
    // Calculate the average of the remaining N-2 runs
    const int trimmed_n = N_RUNS - 2;
    average_time = sum / trimmed_n;
    
    // Calculate Standard Deviation for the trimmed data
    // Sum the squared differences from the mean
    for (i = 1; i < N_RUNS - 1; i++) {
        sum_sq_diff += (times[i] - average_time) * (times[i] - average_time);
    }
    
    // Calculate sample standard deviation (using n-1 denominator)
    if (trimmed_n > 1) {
        std_dev = sqrt(sum_sq_diff / (trimmed_n - 1));
    }


    // 3. Print Results
    printf("\n------------------------------------------------\n");
    printf("        BENCHMARK RESULTS (N=%d)\n", N_RUNS);
    printf("------------------------------------------------\n");
    printf("    Times recorded (s):\n");
    for(i = 0; i < N_RUNS; i++) {
        printf("        [%.6f]%s\n", times[i], (i == 0 || i == N_RUNS - 1) ? " (Trimmed)" : "");
    }
    printf("\n    Trimmed Average Time: %.6f seconds\n", average_time);
    printf("    Standard Deviation (of trimmed): +/- %.6f seconds\n", std_dev);
    printf("------------------------------------------------\n");

    // 4. Export Trimmed Times to CSV
    printf("\n--- Data Export ---\n");
    FILE *csv_file = fopen(CSV_OUTPUT_FILE, "w");
    if (csv_file == NULL) {
        perror("Error creating CSV file");
    } else {
        fprintf(csv_file, "Execution Time (s)\n"); // CSV Header
        // Write the trimmed data (all except the first and last)
        for (i = 1; i < N_RUNS - 1; i++) {
            fprintf(csv_file, "%.6f\n", times[i]);
        }
        fclose(csv_file);
        printf("Trimmed execution times have been exported to %s\n", CSV_OUTPUT_FILE);
    }


    // 5. Image Comparison
    printf("\n--- Output Verification ---\n");
    comparison_result = compare_files(OUTPUT_FILE, REFERENCE_FILE);

    if (comparison_result == 0) {
        printf("Verification: The output file (%s) is **IDENTICAL** to the prototype file (%s).\n", OUTPUT_FILE, REFERENCE_FILE);
    } else if (comparison_result == 1) {
        printf("Verification: The output file (%s) is **DIFFERENT** from the prototype file (%s).\n", OUTPUT_FILE, REFERENCE_FILE);
    } else {
        printf("Verification: Could not complete file comparison due to I/O errors.\n");
        return 1;
    }

    return 0;
}
