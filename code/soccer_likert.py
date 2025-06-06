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

def create_soccer_likert(df):
    """【1-3】サッカーは好きだ（ヒストグラム・箱ひげ図）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    soccer_row = df.iloc[2, 1:]  # 行3（index=2）、列1以降
    
    soccer_values = [x for x in soccer_row.values if pd.notna(x) and str(x).strip() != '']
    soccer_scores = pd.to_numeric(soccer_values, errors='coerce')
    soccer_scores = soccer_scores[~pd.isna(soccer_scores)]  # NaNを除去
    
    if len(soccer_scores) > 0:
        mean_score = soccer_scores.mean()
        median_score = np.median(soccer_scores)
        std_score = soccer_scores.std()
        min_score = soccer_scores.min()
        max_score = soccer_scores.max()
        
        ax1.hist(soccer_scores, bins=5, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=1.5)
        ax1.set_title(f'[3]サッカーは好きだ - ヒストグラム\n(Soccer Interest - Histogram)\n'
                     f'平均: {mean_score:.1f}, 中央値: {median_score:.1f}, 標準偏差: {std_score:.1f}', 
                     fontsize=14, pad=15)
        ax1.set_xlabel('評価スコア (Rating Score)', fontsize=12)
        ax1.set_ylabel('人数 (Count)', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.set_xticklabels(['1\n(全く好きでない)', '2\n(あまり好きでない)', '3\n(どちらでもない)', 
                            '4\n(やや好き)', '5\n(とても好き)'], fontsize=10)
        
        ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'平均: {mean_score:.1f}')
        ax1.axvline(median_score, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'中央値: {median_score:.1f}')
        ax1.legend(fontsize=11)
        
        box_plot = ax2.boxplot(soccer_scores, patch_artist=True, labels=['サッカー好き度'])
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][0].set_alpha(0.7)
        
        ax2.set_title(f'[3]サッカーは好きだ - 箱ひげ図\n(Soccer Interest - Box Plot)\n'
                     f'範囲: {min_score:.0f}-{max_score:.0f}, サンプル数: {len(soccer_scores)}', 
                     fontsize=14, pad=15)
        ax2.set_ylabel('評価スコア (Rating Score)', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_yticks([1, 2, 3, 4, 5])
        ax2.set_yticklabels(['1\n(全く好きでない)', '2\n(あまり好きでない)', '3\n(どちらでもない)', 
                            '4\n(やや好き)', '5\n(とても好き)'], fontsize=10)
        
        y_positions = soccer_scores
        x_positions = np.random.normal(1, 0.04, size=len(soccer_scores))  # 少しジッターを追加
        ax2.scatter(x_positions, y_positions, alpha=0.6, color='darkgreen', s=30, zorder=3)
        
        plt.tight_layout()
        plt.savefig('Visitors/soccer_likert.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ サッカー好きLikert分析保存: Visitors/soccer_likert.png")
        print(f"サッカー好き統計: 平均={mean_score:.1f}, 中央値={median_score:.1f}, 標準偏差={std_score:.1f}")
        print(f"スコア範囲: {min_score:.0f}-{max_score:.0f}, サンプル数: {len(soccer_scores)}")
        
        print("\nサッカー好き度分布:")
        for score in [1, 2, 3, 4, 5]:
            count = sum(soccer_scores == score)
            percentage = (count / len(soccer_scores)) * 100
            print(f"  スコア{score}: {count}名 ({percentage:.1f}%)")
        
        soccer_series = pd.Series(soccer_scores)
        return soccer_series.describe()
    else:
        print("サッカー好きデータが見つかりません")
        return None

def main():
    """メイン実行関数"""
    print("=== [3]サッカーは好きだ Likert分析作成 ===")
    
    setup_japanese_font()
    df = load_data()
    
    print("\nサッカー好きLikert分析の作成...")
    soccer_stats = create_soccer_likert(df)
    
    if soccer_stats is not None:
        print(f"\n詳細統計:\n{soccer_stats}")
    
    print("\n=== サッカー好きLikert分析作成完了 ===")

if __name__ == "__main__":
    main()
