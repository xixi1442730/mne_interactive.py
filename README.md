# mne_interactive.py
我在git hub上的第一个仓库
"""
基于 MNE-Python GitHub 版本的简单交互式处理
修复电极位置问题，保持原始 GitHub 功能
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from mne.preprocessing import ICA

print(f"🎯 MNE-Python 版本: {mne.__version__}")

def find_edf_files():
    """查找 EDF 文件"""
    files = glob.glob("*.edf")
    return sorted(list(set(files)))

def show_file_menu():
    """显示文件选择菜单"""
    files = find_edf_files()
    
    if not files:
        print("❌ 没有找到 EDF 文件！")
        return None
    
    print(f"\n📁 找到 {len(files)} 个 EDF 文件:")
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f) / 1024
        print(f"   {i}. {f} ({size:.0f} KB)")
    
    while True:
        try:
            choice = input(f"\n请选择文件 (1-{len(files)}), 或输入 0 退出: ").strip()
            if choice == '0':
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(files):
                return files[choice_num - 1]
            else:
                print(f"请输入 1-{len(files)} 的数字")
        except ValueError:
            print("请输入有效数字")

def show_processing_menu(filename):
    """显示处理选项菜单"""
    print(f"\n📋 选择的文件: {filename}")
    print("请选择处理方式:")
    print("1. 完整处理 (滤波 + ICA)")
    print("2. 仅滤波处理")
    print("3. 仅查看数据信息")
    print("4. 选择其他文件")
    print("0. 退出")
    
    while True:
        choice = input("\n请选择 (0-4): ").strip()
        if choice in ['0', '1', '2', '3', '4']:
            return choice
        else:
            print("请输入 0-4 的数字")

def load_and_clean_data(filename):
    """加载并清理数据 - GitHub MNE 原始方式"""
    print(f"\n📁 加载数据: {filename}")
    raw = mne.io.read_raw_edf(filename, preload=True)
    
    # 修复通道名称 - 去除重复和点号
    channel_mapping = {}
    seen_channels = set()
    unique_suffix = 0
    
    for ch_name in raw.ch_names:
        new_name = ch_name.rstrip('.').rstrip()
        
        # 处理重复通道名
        if new_name in seen_channels:
            new_name = f"{new_name}_{unique_suffix}"
            unique_suffix += 1
        else:
            seen_channels.add(new_name)
            
        channel_mapping[ch_name] = new_name
    
    raw.rename_channels(channel_mapping)
    
    # 设置电极位置 - 使用 GitHub 原始方式但跳过重叠检查
    try:
        raw.set_montage('standard_1020', on_missing='ignore')
        print("✅ 电极位置设置完成")
    except Exception as e:
        print(f"⚠ 电极位置设置警告: {e}")
    
    print(f"✅ 数据加载成功")
    print(f"   采样率: {raw.info['sfreq']} Hz")
    print(f"   通道数: {len(raw.ch_names)}")
    print(f"   时长: {raw.times[-1]:.2f} 秒")
    
    return raw

def preprocess_data(raw):
    """预处理 - GitHub MNE 原始方式"""
    print("\n🔧 进行预处理...")
    raw_filtered = raw.copy()
    
    # 带通滤波
    raw_filtered.filter(1.0, 40.0)
    # 陷波滤波
    raw_filtered.notch_filter(50)
    
    print("✅ 滤波完成")
    return raw_filtered

def run_ica_cleaning(raw_filtered):
    """ICA 去噪 - GitHub MNE 原始方式"""
    print("\n🎯 进行 ICA 去噪...")
    
    # 准备 ICA 数据
    raw_for_ica = raw_filtered.copy()
    raw_for_ica.filter(1.0, None)
    
    # 拟合 ICA - GitHub 原始方式
    ica = ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw_for_ica)
    
    # 显示 ICA 源信号 - 这个不受电极位置影响
    print("📊 显示 ICA 源信号...")
    ica.plot_sources(raw_filtered, show=True)
    
    # 自动检测伪迹 - 使用 GitHub 原始函数
    ica.exclude = []
    
    # 检测心电伪迹 - GitHub 原始方式
    print("❤️  检测心电伪迹...")
    try:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filtered)
        print(f"   检测到的心电成分: {ecg_indices}")
        ica.exclude.extend(ecg_indices)
    except Exception as e:
        print(f"   心电检测失败: {e}")
    
    print(f"❌ 排除的成分: {ica.exclude}")
    
    # 手动选择 - 保持交互
    if not ica.exclude:
        print("💡 自动检测未找到明显的伪迹成分")
        manual_input = input("是否手动输入要排除的成分编号？(y/n): ").strip().lower()
        if manual_input in ['y', 'yes']:
            try:
                comps = input("请输入要排除的成分编号（用逗号分隔，如: 0,1,4）: ").strip()
                ica.exclude = [int(x.strip()) for x in comps.split(',') if x.strip()]
                print(f"✅ 手动排除成分: {ica.exclude}")
            except:
                print("❌ 输入格式错误，跳过手动选择")
    
    # 应用 ICA - GitHub 原始方式
    raw_cleaned = raw_filtered.copy()
    ica.apply(raw_cleaned)
    
    print("✅ ICA 去噪完成")
    return raw_cleaned, ica

def show_data_info(raw):
    """显示数据信息"""
    print("\n📊 数据详细信息:")
    print(f"通道: {raw.ch_names}")
    print(f"采样率: {raw.info['sfreq']} Hz")
    print(f"数据点数: {len(raw.times)}")
    
    # 简单预览
    input("\n按回车键查看数据预览...")
    raw.plot(duration=5, title="数据预览")

def compare_results(raw_before, raw_after, title_before, title_after):
    """比较处理前后结果"""
    print(f"\n📈 比较 {title_before} 和 {title_after}...")
    
    # 选择几个通道显示
    chs = raw_before.ch_names[:4] if len(raw_before.ch_names) >= 4 else raw_before.ch_names
    
    raw_before.plot(title=title_before, picks=chs, block=False)
    raw_after.plot(title=title_after, picks=chs, block=True)

def save_results(raw, filename, prefix):
    """保存结果"""
    output_file = f"{prefix}_{filename.replace('.edf', '.fif')}"
    raw.save(output_file, overwrite=True)
    print(f"💾 保存: {output_file}")
    return output_file

# 主程序
def main():
    print("="*50)
    print("MNE-Python EEG 处理系统")
    print("="*50)
    
    current_file = None
    current_raw = None
    current_filtered = None
    
    while True:
        if current_file is None:
            current_file = show_file_menu()
            if current_file is None:
                break
            current_raw = load_and_clean_data(current_file)
        
        choice = show_processing_menu(current_file)
        
        if choice == '0':  # 退出
            break
        elif choice == '1':  # 完整处理
            current_filtered = preprocess_data(current_raw)
            current_cleaned, ica = run_ica_cleaning(current_filtered)
            
            # 比较结果
            compare_results(current_filtered, current_cleaned, "ICA前", "ICA后")
            
            # 保存结果
            save_results(current_filtered, current_file, "filtered")
            save_results(current_cleaned, current_file, "ica_cleaned")
            
            print(f"\n🎉 {current_file} 完整处理完成！")
            
        elif choice == '2':  # 仅滤波
            current_filtered = preprocess_data(current_raw)
            compare_results(current_raw, current_filtered, "原始数据", "滤波后")
            save_results(current_filtered, current_file, "filtered")
            print(f"\n✅ {current_file} 滤波处理完成！")
            
        elif choice == '3':  # 仅查看信息
            show_data_info(current_raw)
            
        elif choice == '4':  # 选择其他文件
            current_file = None
            current_raw = None
            current_filtered = None
    
    print("\n👋 感谢使用 MNE-Python！")

if __name__ == "__main__":
    main()
