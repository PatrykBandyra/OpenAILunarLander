import simulation
import random_approach
import hard_coded_approach

"""
This file contains simulations of all pre-trained models used in experiments.
"""

if __name__ == '__main__':
    print("Simulation of random approach\n")
    random_approach.main()
    print('\n')

    print("Simulation of hard coded approach\n")
    hard_coded_approach.main()
    print('\n')

    print("Simulation of our model\n")
    simulation.main()
    print("\n")

    print("Simulation of our model with dropout\n")
    simulation.main(model_name="ModelDropout", model_suffix='_dropout')
    print("\n")

    print("Simulation of our model with L1Loss function\n")
    simulation.main(model_name="ModelL1Loss", model_suffix='_L1Loss')
    print("\n")

    print("Simulation of our model with too big learning rate\n")
    simulation.main(model_suffix='_LR0002', lr=0.002)
    print("\n")

    print("Simulation of our model with too small learning rate\n")
    simulation.main(model_suffix='_LR00004', lr=0.0004)
    print("\n")
