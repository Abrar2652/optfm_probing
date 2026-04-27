import os
import re
import pickle

def process_files(input_dir, output_dir):
    """
    修改指定目录中所有符合格式的pickle文件中的label属性，并保存到新目录。
    
    参数:
        input_dir (str): 输入目录路径。
        output_dir (str): 输出目录路径。
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义正则表达式来匹配文件名模式
    file_pattern = re.compile(r'^sample_(\d+)_(\d+)\.pkl$')

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        match = file_pattern.match(filename)
        if match:
            a, b = map(int, match.groups())
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # 读取pickle文件
                with open(input_path, 'rb') as f:
                    data_dict = pickle.load(f)

                # 修改label属性
                data_dict['label'] = a % 317

                # 将修改后的数据保存到新目录
                with open(output_path, 'wb') as f:
                    pickle.dump(data_dict, f)

                print(f"Processed and saved: {output_path}")

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "/ml_nfs/samples"
    output_directory = "/ml_nfs/samples_1"
    
    process_files(input_directory, output_directory)