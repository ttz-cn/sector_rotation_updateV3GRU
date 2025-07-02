import subprocess
import argparse
import os
from datetime import datetime
import sys
from predict_gru_model import predict_returns_for_date
from run_optimizer import run_optimization


def run_gru_factor_fusion():
    """
    运行GRU因子融合模型的训练和预测
    """
    parser = argparse.ArgumentParser(description="运行GRU因子融合模型")
    parser.add_argument("--train", action="store_true", help="是否重新训练模型")
    parser.add_argument("--predict", action="store_true", help="是否进行预测")
    parser.add_argument("--optimize", action="store_true", help="是否进行投资组合优化")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # 记录开始时间
    start_time = datetime.now()
    print(f"开始时间: {start_time}")

    # 记录运行日志
    log_file = f"./output/gru_run_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"

    # 创建日志文件
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"GRU因子融合模型运行日志\n")
        f.write(f"开始时间: {start_time}\n")
        f.write(f"参数: train={args.train}, predict={args.predict}, optimize={args.optimize}\n\n")

    try:
        # 训练模型
        if args.train:
            print("\n开始训练GRU因子融合模型...")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n开始训练GRU因子融合模型...\n")

            # 调用训练脚本
            train_result = subprocess.run([sys.executable, "train_gru_model.py"], capture_output=True, text=True)

            # 记录训练结果
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("训练输出:\n")
                f.write(train_result.stdout)
                if train_result.stderr:
                    f.write("\n训练错误:\n")
                    f.write(train_result.stderr)

            print("训练完成")

        # 预测
        if args.predict:
            print("\n开始使用GRU模型进行预测...")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n开始使用GRU模型进行预测...\n")

            # 构建预测命令
            predict_cmd = [sys.executable, "predict_gru_model.py"]
            if args.optimize:
                predict_cmd.append("--optimize")

            # 调用预测脚本
            predict_result = subprocess.run(predict_cmd, capture_output=True, text=True)

            # 记录预测结果
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("预测输出:\n")
                f.write(predict_result.stdout)
                if predict_result.stderr:
                    f.write("\n预测错误:\n")
                    f.write(predict_result.stderr)

            print("预测完成")

        # 记录结束时间
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n结束时间: {end_time}")
        print(f"总运行时间: {duration}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n结束时间: {end_time}\n")
            f.write(f"总运行时间: {duration}\n")

    except Exception as e:
        error_msg = f"运行过程中发生错误: {str(e)}"
        print(error_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{error_msg}\n")
        raise e


if __name__ == "__main__":
    run_gru_factor_fusion()
