"""Script for training env generator"""
from highrl.utils.logger import init_logger
from highrl.envs.env_generator import main

if __name__ == "__main__":
    init_logger()
    main()
