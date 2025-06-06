import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import pearsonr
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

def create_soccer_vs_tech_scatter(df):
    """【2-1】サッカーは好きだ vs ARやVRなど...親しみがある（散布図＋相関係数）"""
    plt.figure(figsize=(12, 8))
    
    soccer_row = df.iloc[2, 1:]
    soccer_values = [x for x in soccer_row.values if pd.notna(x) and str(x).strip() != '']
    soccer_scores = pd.to_numeric(soccer_values, errors='coerce')
    soccer_scores = soccer_scores[~pd.isna(soccer_scores)]
    
    tech_row = df.iloc[3, 1:]
    tech_values = [x for x in tech_row.values if pd.notna(x) and str(x).strip() != '']
    tech_scores = pd.to_numeric(tech_values, errors='coerce')
    tech_scores = tech_scores[~pd.isna(tech_scores)]
    
    min_length = min(len(soccer_scores), len(tech_scores))
    soccer_data = soccer_scores[:min_length]
    tech_data = tech_scores[:min_length]
    
    if len(soccer_data) > 0 and len(tech_data) > 0:
        correlation, p_value = pearsonr(soccer_data, tech_data)
        
        plt.scatter(soccer_data, tech_data, alpha=0.7, s=80, color='purple', edgecolors='black', linewidth=1)
        
        z = np.polyfit(soccer_data, tech_data, 1)
        p = np.poly1d(z)
        plt.plot(soccer_data, p(soccer_data), "r--", alpha=0.8, linewidth=2)
        
        soccer_mean = soccer_data.mean()
        tech_mean = tech_data.mean()
        
        plt.title(f'[3]サッカーは好きだ vs [4]ARやVRなどデジタルテクノロジーに親しみがある\n'
                 f'(Soccer Interest vs Digital Technology Affinity)\n'
                 f'相関係数: r = {correlation:.3f} (p = {p_value:.3f}), サンプル数: {len(soccer_data)}', 
                 fontsize=14, pad=20)
        
        plt.xlabel('[3]サッカーは好きだ (Soccer Interest)\n'
                  f'平均: {soccer_mean:.1f}', fontsize=12)
        plt.ylabel('[4]ARやVRなどデジタルテクノロジーに親しみがある\n'
                  f'(Digital Technology Affinity) 平均: {tech_mean:.1f}', fontsize=12)
        
        plt.xlim(0.5, 5.5)
        plt.ylim(0.5, 5.5)
        plt.xticks([1, 2, 3, 4, 5], ['1\n(全く好きでない)', '2\n(あまり好きでない)', '3\n(どちらでもない)', 
                                    '4\n(やや好き)', '5\n(とても好き)'], fontsize=10)
        plt.yticks([1, 2, 3, 4, 5], ['1\n(全く親しみがない)', '2\n(あまり親しみがない)', '3\n(どちらでもない)', 
                                    '4\n(やや親しみがある)', '5\n(とても親しみがある)'], fontsize=10)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        
        if abs(correlation) >= 0.7:
            correlation_strength = "強い相関"
        elif abs(correlation) >= 0.4:
            correlation_strength = "中程度の相関"
        elif abs(correlation) >= 0.2:
            correlation_strength = "弱い相関"
        else:
            correlation_strength = "ほとんど相関なし"
        
        plt.text(0.02, 0.98, f'相関の強さ: {correlation_strength}', 
                transform=plt.gca().transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('Visitors/soccer_vs_tech_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ サッカー好き vs テクノロジー親和性散布図保存: Visitors/soccer_vs_tech_scatter.png")
        print(f"相関分析結果: r = {correlation:.3f}, p = {p_value:.3f}")
        print(f"サッカー好き平均: {soccer_mean:.1f}, テクノロジー親和性平均: {tech_mean:.1f}")
        print(f"相関の強さ: {correlation_strength}")
        print(f"サンプル数: {len(soccer_data)}")
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'sample_size': len(soccer_data),
            'soccer_mean': soccer_mean,
            'tech_mean': tech_mean,
            'correlation_strength': correlation_strength
        }
    else:
        print("サッカー好きまたはテクノロジー親和性データが見つかりません")
        return None

def main():
    """メイン実行関数"""
    print("=== [3]サッカーは好きだ vs [4]ARやVRなどデジタルテクノロジーに親しみがある 相関分析 ===")
    
    setup_japanese_font()
    df = load_data()
    
    print("\nサッカー好き vs テクノロジー親和性相関分析の作成...")
    correlation_stats = create_soccer_vs_tech_scatter(df)
    
    if correlation_stats is not None:
        print(f"\n相関分析完了:")
        print(f"  相関係数: {correlation_stats['correlation']:.3f}")
        print(f"  有意性: p = {correlation_stats['p_value']:.3f}")
        print(f"  相関の強さ: {correlation_stats['correlation_strength']}")
    
    print("\n=== サッカー好き vs テクノロジー親和性相関分析完了 ===")

if __name__ == "__main__":
    main()
