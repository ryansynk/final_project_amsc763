import numpy as np
import matplotlib.pyplot as plt

def main():
    cml_gpu_times = np.array([0.038031,
                              0.149178,
                              0.307572,
                              0.693915,
                              1.221767,
                              1.964808,
                              3.163227,
                              5.080235,
                              7.457819,
                              10.617740,
                              14.746815])

    python_cpu_times = np.array([0.0870554447174072,
                                 0.3439326286315918,
                                 0.8373477458953857,
                                 1.7350046634674072,
                                 3.0339972972869873,
                                 5.112581729888916 ,
                                 7.464255094528198 ,
                                 11.354997873306274,
                                 15.293460845947266,
                                 20.04918646812439 ,
                                 26.13704013824463 ,
                                 35.886847734451294,
                                 41.95623207092285 ])

    matrix_rows = np.array([640, 
                            1280,
                            1920,
                            2560,
                            3200,
                            3840,
                            4480,
                            5120,
                            5760,
                            6400,
                            7040,
                            7680,
                            8320])

    speedup = python_cpu_times[:-2] / cml_gpu_times

    fig, ax = plt.subplots()
    ax.plot(matrix_rows[:-2], python_cpu_times[:-2], "x-", label="CPU")
    ax.plot(matrix_rows[:-2], cml_gpu_times, "o-", label="GPU")
    ax.set_title("Runtime vs Matrix Size")
    ax.set_xlabel("Matrix rows")
    ax.set_ylabel("Runtime (s)")
    ax.legend()
    plt.savefig("python_cpu_size_v_runtime_and_gpu")

    fig2, ax2 = plt.subplots()
    ax2.plot(matrix_rows[:-2], cml_gpu_times, "x-")
    ax2.set_title("Runtime vs Matrix Size, CUDA GPU Implementation")
    ax2.set_xlabel("Matrix rows")
    ax2.set_ylabel("Runtime (s)")
    plt.savefig("cuda_gpu_size_v_runtime")

    fig3, ax3 = plt.subplots()
    ax3.plot(matrix_rows[:-2], speedup, "x-")
    ax2.set_title("Speedup of GPU over CPU runtime")
    ax2.set_xlabel("Matrix rows")
    ax2.set_ylabel("Speedup")
    plt.savefig("gpu_cpu_speedup")


if __name__ == "__main__":
    main()
