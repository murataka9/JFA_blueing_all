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

def create_tech_vs_exhibition_scatter(df):
    """【2-3】ARやVRなど...親しみがある vs デジタル・テクノロジー展示は好きだ（散布図＋相関係数）"""
    plt.figure(figsize=(12, 8))
    
    tech_row = df.iloc[3, 1:]
    tech_values = [x for x in tech_row.values if pd.notna(x) and str(x).strip() != '']
    tech_scores = pd.to_numeric(tech_values, errors='coerce')
    tech_scores = tech_scores[~pd.isna(tech_scores)]
    
    exhibition_row = df.iloc[4, 1:]
    exhibition_values = [x for x in exhibition_row.values if pd.notna(x) and str(x).strip() != '']
    exhibition_scores = pd.to_numeric(exhibition_values, errors='coerce')
    exhibition_scores = exhibition_scores[~pd.isna(exhibition_scores)]
    
    min_length = min(len(tech_scores), len(exhibition_scores))
    tech_data = tech_scores[:min_length]
    exhibition_data = exhibition_scores[:min_length]
    
    if len(tech_data) > 0 and len(exhibition_data) > 0:
        correlation, p_value = pearsonr(tech_data, exhibition_data)
        
        plt.scatter(tech_data, exhibition_data, alpha=0.7, s=80, color='green', edgecolors='black', linewidth=1)
        
        z = np.polyfit(tech_data, exhibition_data, 1)
        p = np.poly1d(z)
        plt.plot(tech_data, p(tech_data), "r--", alpha=0.8, linewidth=2)
        
        tech_mean = tech_data.mean()
        exhibition_mean = exhibition_data.mean()
        
        plt.title(f'[4]ARやVRなどデジタルテクノロジーに親しみがある vs [5]デジタル・テクノロジー展示は好きだ\n'
                 f'(Digital Technology Affinity vs Digital Technology Exhibition Preference)\n'
                 f'相関係数: r = {correlation:.3f} (p = {p_value:.3f}), サンプル数: {len(tech_data)}', 
                 fontsize=14, pad=20)
        
        plt.xlabel('[4]ARやVRなどデジタルテクノロジーに親しみがある\n'
                  f'(Digital Technology Affinity) 平均: {tech_mean:.1f}', fontsize=12)
        plt.ylabel('[5]デジタル・テクノロジー展示は好きだ\n'
                  f'(Digital Technology Exhibition Preference) 平均: {exhibition_mean:.1f}', fontsize=12)
        
        plt.xlim(0.5, 5.5)
        plt.ylim(0.5, 5.5)
        plt.xticks([1, 2, 3, 4, 5], ['1\n(全く親しみがない)', '2\n(あまり親しみがない)', '3\n(どちらでもない)', 
                                    '4\n(やや親しみがある)', '5\n(とても親しみがある)'], fontsize=10)
        plt.yticks([1, 2, 3, 4, 5], ['1\n(全く好きでない)', '2\n(あまり好きでない)', '3\n(どちらでもない)', 
                                    '4\n(やや好き)', '5\n(とても好き)'], fontsize=10)
        
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
        plt.savefig('Visitors/tech_vs_exhibition_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ テクノロジー親和性 vs デジタル展示好み散布図保存: Visitors/tech_vs_exhibition_scatter.png")
        print(f"相関分析結果: r = {correlation:.3f}, p = {p_value:.3f}")
        print(f"テクノロジー親和性平均: {tech_mean:.1f}, デジタル展示好み平均: {exhibition_mean:.1f}")
        print(f"相関の強さ: {correlation_strength}")
        print(f"サンプル数: {len(tech_data)}")
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'sample_size': len(tech_data),
            'tech_mean': tech_mean,
            'exhibition_mean': exhibition_mean,
            'correlation_strength': correlation_strength
        }
    else:
        print("テクノロジー親和性またはデジタル展示好みデータが見つかりません")
        return None

def main():
    """メイン実行関数"""
    print("=== [4]ARやVRなどデジタルテクノロジーに親しみがある vs [5]デジタル・テクノロジー展示は好きだ 相関分析 ===")
    
    setup_japanese_font()
    df = load_data()
    
    print("\nテクノロジー親和性 vs デジタル展示好み相関分析の作成...")
    correlation_stats = create_tech_vs_exhibition_scatter(df)
    
    if correlation_stats is not None:
        print(f"\n相関分析完了:")
        print(f"  相関係数: {correlation_stats['correlation']:.3f}")
        print(f"  有意性: p = {correlation_stats['p_value']:.3f}")
        print(f"  相関の強さ: {correlation_stats['correlation_strength']}")
    
    print("\n=== テクノロジー親和性 vs デジタル展示好み相関分析完了 ===")

if __name__ == "__main__":
    main()
