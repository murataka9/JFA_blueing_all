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

def create_age_histogram(df):
    """【1-1】年齢のヒストグラム"""
    plt.figure(figsize=(12, 8))
    
    age_row = df.iloc[0, 1:]  # 行1（index=0）、列1以降
    
    age_values = [x for x in age_row.values if pd.notna(x) and str(x).strip() != '']
    ages = pd.to_numeric(age_values, errors='coerce')
    ages = ages[~pd.isna(ages)]  # NaNを除去
    
    if len(ages) > 0:
        plt.hist(ages, bins=8, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
        
        mean_age = ages.mean()
        median_age = np.median(ages)
        std_age = ages.std()
        min_age = ages.min()
        max_age = ages.max()
        
        plt.title(f'[1]ご年齢の分布 (Age Distribution)\n'
                 f'平均: {mean_age:.1f}歳, 中央値: {median_age:.1f}歳, 標準偏差: {std_age:.1f}歳\n'
                 f'範囲: {min_age:.0f}-{max_age:.0f}歳 (N={len(ages)})', 
                 fontsize=16, pad=20)
        
        plt.xlabel('年齢 (Age)', fontsize=14)
        plt.ylabel('人数 (Count)', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'平均: {mean_age:.1f}歳')
        plt.axvline(median_age, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'中央値: {median_age:.1f}歳')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('Visitors/age_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 年齢ヒストグラム保存: Visitors/age_histogram.png")
        print(f"年齢統計: 平均={mean_age:.1f}, 中央値={median_age:.1f}, 標準偏差={std_age:.1f}")
        print(f"年齢範囲: {min_age:.0f}-{max_age:.0f}歳, サンプル数: {len(ages)}")
        
        ages_series = pd.Series(ages)
        return ages_series.describe()
    else:
        print("年齢データが見つかりません")
        return None

def main():
    """メイン実行関数"""
    print("=== [1]ご年齢のヒストグラム作成 ===")
    
    setup_japanese_font()
    df = load_data()
    
    print("\n年齢分布の作成...")
    age_stats = create_age_histogram(df)
    
    if age_stats is not None:
        print(f"\n詳細統計:\n{age_stats}")
    
    print("\n=== 年齢ヒストグラム作成完了 ===")

if __name__ == "__main__":
    main()
