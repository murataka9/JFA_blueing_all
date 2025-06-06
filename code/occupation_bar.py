import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

def setup_japanese_font():
    """日本語フォントの設定"""
    try:
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print("✓ 日本語フォント設定完了")
        return True
    except Exception as e:
        print(f"日本語フォント設定エラー: {e}")
        return False

def load_data():
    """データの読み込み"""
    df = pd.read_csv('Visitors/visitors survey_data.csv')
    print(f"データ形状: {df.shape}")
    return df

def create_occupation_bar(df):
    """【1-2】職種の棒グラフ（頻度）"""
    plt.figure(figsize=(14, 8))
    
    occupation_row = df.iloc[1, 1:]  # 行2（index=1）、列1以降
    
    occupation_values = [str(x) for x in occupation_row.values if pd.notna(x) and str(x).strip() != '']
    
    if len(occupation_values) > 0:
        occupation_counts = pd.Series(occupation_values).value_counts()
        
        bars = plt.bar(range(len(occupation_counts)), occupation_counts.values, 
                      alpha=0.7, color='lightcoral', edgecolor='black', linewidth=1.5)
        
        total_responses = len(occupation_values)
        unique_occupations = len(occupation_counts)
        most_common = occupation_counts.index[0]
        most_common_count = occupation_counts.iloc[0]
        
        plt.title(f'[2]ご職種の分布 (Occupation Distribution)\n'
                 f'総回答数: {total_responses}名, 職種数: {unique_occupations}種類\n'
                 f'最多職種: {most_common} ({most_common_count}名)', 
                 fontsize=16, pad=20)
        
        plt.xticks(range(len(occupation_counts)), occupation_counts.index, 
                  rotation=45, ha='right', fontsize=12)
        plt.ylabel('人数 (Count)', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('Visitors/occupation_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 職種棒グラフ保存: Visitors/occupation_bar.png")
        print(f"職種統計: 総回答数={total_responses}, 職種数={unique_occupations}")
        print(f"最多職種: {most_common} ({most_common_count}名)")
        
        print("\n職種別分布:")
        for occupation, count in occupation_counts.items():
            percentage = (count / total_responses) * 100
            print(f"  {occupation}: {count}名 ({percentage:.1f}%)")
        
        return occupation_counts
    else:
        print("職種データが見つかりません")
        return None

def main():
    """メイン実行関数"""
    print("=== [2]ご職種の棒グラフ作成 ===")
    
    setup_japanese_font()
    df = load_data()
    
    print("\n職種分布の作成...")
    occupation_stats = create_occupation_bar(df)
    
    if occupation_stats is not None:
        print(f"\n職種分布完了: {len(occupation_stats)}種類の職種を可視化")
    
    print("\n=== 職種棒グラフ作成完了 ===")

if __name__ == "__main__":
    main()
