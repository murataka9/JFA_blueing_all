import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager as fm
from scipy import stats

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Takao Gothic', 'IPAex Gothic', 'DejaVu Sans']

def load_and_clean_data():
    df = pd.read_csv('Visitors/visitors survey_data.csv')
    
    df = df.T
    df.columns = df.iloc[0]  # Use first row as column names
    df = df.drop(df.index[0])  # Remove the header row
    
    df.reset_index(drop=True, inplace=True)
    
    column_mapping = {
        '[1]ご年齢': '年齢 (Age)',
        '[3]サッカーは好きだ': 'サッカーへの関心 (Interest in Soccer)',
        '[4]ARやVRなどデジタルテクノロジーに親しみがある': 'テクノロジーへの親和性 (Technology Affinity)',
        '[5]デジタル・テクノロジー展示は好きだ': 'デジタル展示への嗜好 (Digital Exhibition Preference)',
        '[2]DIGITAL COLLECTION展示の満足度について教えて下さい。': 'デジタル展示満足度 (Digital Exhibition Satisfaction)',
        '[3]ROAD TO 2050などの実物展示を使った展示の満足度について教えて下さい。': '実物展示満足度 (Traditional Exhibition Satisfaction)'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    numeric_columns = ['年齢 (Age)', 'サッカーへの関心 (Interest in Soccer)', 
                      'テクノロジーへの親和性 (Technology Affinity)', 
                      'デジタル展示への嗜好 (Digital Exhibition Preference)',
                      'デジタル展示満足度 (Digital Exhibition Satisfaction)',
                      '実物展示満足度 (Traditional Exhibition Satisfaction)']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(how='all')
    
    return df

def create_demographics_plot(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    age_col = '年齢 (Age)'
    soccer_col = 'サッカーへの関心 (Interest in Soccer)'
    tech_col = 'テクノロジーへの親和性 (Technology Affinity)'
    digital_pref_col = 'デジタル展示への嗜好 (Digital Exhibition Preference)'
    
    ax1.hist(df[age_col].dropna(), bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('年齢分布 (Age Distribution)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('年齢 (Age)', fontsize=12)
    ax1.set_ylabel('人数 (Count)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(df[soccer_col].dropna(), bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title('サッカーへの関心度 (Interest in Soccer)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('関心度 (1-5)', fontsize=12)
    ax2.set_ylabel('人数 (Count)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    ax3.hist(df[tech_col].dropna(), bins=5, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_title('テクノロジーへの親和性 (Technology Affinity)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('親和性 (1-5)', fontsize=12)
    ax3.set_ylabel('人数 (Count)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    ax4.hist(df[digital_pref_col].dropna(), bins=5, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_title('デジタル展示への嗜好 (Digital Exhibition Preference)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('嗜好度 (1-5)', fontsize=12)
    ax4.set_ylabel('人数 (Count)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Visitors/demographics_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_satisfaction_comparison(df):
    digital_col = 'デジタル展示満足度 (Digital Exhibition Satisfaction)'
    traditional_col = '実物展示満足度 (Traditional Exhibition Satisfaction)'
    
    digital_scores = df[digital_col].dropna()
    traditional_scores = df[traditional_col].dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    categories = ['デジタル展示\n(Digital)', '実物展示\n(Traditional)']
    means = [digital_scores.mean(), traditional_scores.mean()]
    stds = [digital_scores.std(), traditional_scores.std()]
    
    bars = ax1.bar(categories, means, yerr=stds, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black')
    ax1.set_title('物理展示とSRD展示の満足度比較', fontsize=16, fontweight='bold')
    ax1.set_ylabel('満足度 (1-5)', fontsize=12)
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
    
    x_pos = np.arange(len(digital_scores))
    ax2.scatter(x_pos, digital_scores, alpha=0.7, color='#FF6B6B', s=60, label='デジタル展示 (Digital)')
    ax2.scatter(x_pos, traditional_scores, alpha=0.7, color='#4ECDC4', s=60, label='実物展示 (Traditional)')
    ax2.set_title('個別回答の分布 (Individual Responses)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('回答者番号 (Respondent Number)', fontsize=12)
    ax2.set_ylabel('満足度 (1-5)', fontsize=12)
    ax2.set_ylim(0, 5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Visitors/satisfaction_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_technology_correlation(df):
    tech_col = 'テクノロジーへの親和性 (Technology Affinity)'
    digital_sat_col = 'デジタル展示満足度 (Digital Exhibition Satisfaction)'
    digital_pref_col = 'デジタル展示への嗜好 (Digital Exhibition Preference)'
    
    clean_df = df[[tech_col, digital_sat_col, digital_pref_col]].dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(clean_df[tech_col], clean_df[digital_sat_col], 
               alpha=0.7, color='purple', s=80)
    z1 = np.polyfit(clean_df[tech_col], clean_df[digital_sat_col], 1)
    p1 = np.poly1d(z1)
    ax1.plot(clean_df[tech_col], p1(clean_df[tech_col]), "r--", alpha=0.8)
    
    corr1 = clean_df[tech_col].corr(clean_df[digital_sat_col])
    ax1.set_title(f'テクノロジー親和性 vs デジタル満足度\n(相関係数: {corr1:.3f})', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('テクノロジー親和性 (1-5)', fontsize=12)
    ax1.set_ylabel('デジタル展示満足度 (1-5)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(clean_df[digital_pref_col], clean_df[digital_sat_col], 
               alpha=0.7, color='orange', s=80)
    z2 = np.polyfit(clean_df[digital_pref_col], clean_df[digital_sat_col], 1)
    p2 = np.poly1d(z2)
    ax2.plot(clean_df[digital_pref_col], p2(clean_df[digital_pref_col]), "r--", alpha=0.8)
    
    corr2 = clean_df[digital_pref_col].corr(clean_df[digital_sat_col])
    ax2.set_title(f'デジタル嗜好 vs デジタル満足度\n(相関係数: {corr2:.3f})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('デジタル展示への嗜好 (1-5)', fontsize=12)
    ax2.set_ylabel('デジタル展示満足度 (1-5)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Visitors/technology_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_radar(df):
    """Create separate box plots for AR/VR experience comparison [a][b][c][d] with statistical significance tests"""
    
    aspects = {
        '[a]見た目がリアル (Visual Realism)': '[a]見た目がリアル',
        '[b]体験が楽しい (Experience Enjoyment)': '[b]体験が楽しい', 
        '[c]操作がしやすい (Ease of Operation)': '[c]操作がしやすい',
        '[d]手軽だった (Convenience)': '[d]手軽だった'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, (aspect_name, col_name) in enumerate(aspects.items()):
        if col_name in df.columns:
            data = pd.to_numeric(df[col_name], errors='coerce').dropna()
            
            if len(data) > 0:
                bp = axes[i].boxplot([data], patch_artist=True, tick_labels=['AR/VR比較評価\n(vs AR/VR Experience)'])
                bp['boxes'][0].set_facecolor(colors[i])
                bp['boxes'][0].set_alpha(0.7)
                
                y_jitter = np.random.normal(1, 0.04, size=len(data))
                axes[i].scatter(y_jitter, data, alpha=0.6, color=colors[i], s=40)
                
                axes[i].set_ylabel('評価スコア (Rating Score)', fontsize=12)
                axes[i].set_title(f'{aspect_name}\n(デジタル展示 vs AR/VR体験)', 
                                fontsize=14, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(0.5, 5.5)
                
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()
                
                t_stat, p_value = stats.ttest_1samp(data, 3.0)
                
                if p_value < 0.001:
                    sig_text = "***"
                    sig_level = "p<0.001"
                elif p_value < 0.01:
                    sig_text = "**"
                    sig_level = "p<0.01"
                elif p_value < 0.05:
                    sig_text = "*"
                    sig_level = "p<0.05"
                else:
                    sig_text = "n.s."
                    sig_level = f"p={p_value:.3f}"
                
                cohens_d = (mean_val - 3.0) / std_val if std_val > 0 else 0
                
                axes[i].text(0.7, 5.2, f'平均: {mean_val:.2f}\n中央値: {median_val:.2f}\n標準偏差: {std_val:.2f}\nN={len(data)}\n\nt検定 vs 3.0:\nt={t_stat:.2f}, {sig_level}\n効果量d={cohens_d:.2f} {sig_text}\n\n※高スコア=デジタル展示が\nAR/VR体験より優秀', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                print(f"{aspect_name}: Mean={mean_val:.2f}, t={t_stat:.2f}, p={p_value:.3f} {sig_text}, Cohen's d={cohens_d:.2f}, N={len(data)}")
            else:
                axes[i].text(0.5, 0.5, 'データなし\n(No Data)', ha='center', va='center', 
                           transform=axes[i].transAxes, fontsize=14)
                axes[i].set_title(f'{aspect_name}\n(データなし)', fontsize=14, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f'列が見つかりません\n(Column not found)\n{col_name}', 
                        ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'{aspect_name}\n(列なし)', fontsize=14, fontweight='bold')
    
    plt.suptitle('AR/VR経験者によるデジタル展示評価（統計的有意性検定付き）\n(Digital Exhibition Evaluation by AR/VR Experienced Users with Statistical Tests)', 
                fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('Visitors/comparison_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Individual comparison aspect box plots with significance tests saved: comparison_radar.png")

def print_summary_statistics(df):
    print("=== 来場者調査データ分析結果 (Visitor Survey Analysis Results) ===\n")
    
    age_col = '年齢 (Age)'
    soccer_col = 'サッカーへの関心 (Interest in Soccer)'
    tech_col = 'テクノロジーへの親和性 (Technology Affinity)'
    digital_pref_col = 'デジタル展示への嗜好 (Digital Exhibition Preference)'
    digital_sat_col = 'デジタル展示満足度 (Digital Exhibition Satisfaction)'
    traditional_sat_col = '実物展示満足度 (Traditional Exhibition Satisfaction)'
    
    print(f"回答者数 (Total Respondents): {len(df)}")
    
    if age_col in df.columns:
        age_data = df[age_col].dropna()
        if len(age_data) > 0:
            print(f"平均年齢 (Average Age): {age_data.mean():.1f}歳 ({age_data.min():.0f}-{age_data.max():.0f}歳)")
    
    if soccer_col in df.columns:
        soccer_data = df[soccer_col].dropna()
        if len(soccer_data) > 0:
            print(f"サッカー関心度 (Soccer Interest): {soccer_data.mean():.1f}/5")
    
    if tech_col in df.columns:
        tech_data = df[tech_col].dropna()
        if len(tech_data) > 0:
            print(f"テクノロジー親和性 (Technology Affinity): {tech_data.mean():.1f}/5")
    
    if digital_pref_col in df.columns:
        pref_data = df[digital_pref_col].dropna()
        if len(pref_data) > 0:
            print(f"デジタル展示嗜好 (Digital Exhibition Preference): {pref_data.mean():.1f}/5")
    
    print()
    
    digital_scores = df[digital_sat_col].dropna() if digital_sat_col in df.columns else pd.Series()
    traditional_scores = df[traditional_sat_col].dropna() if traditional_sat_col in df.columns else pd.Series()
    
    print("=== 満足度比較 (Satisfaction Comparison) ===")
    if len(digital_scores) > 0:
        print(f"デジタル展示満足度 (Digital): {digital_scores.mean():.1f}/5 ({len(digital_scores)}名)")
    if len(traditional_scores) > 0:
        print(f"実物展示満足度 (Traditional): {traditional_scores.mean():.1f}/5 ({len(traditional_scores)}名)")
    
    if len(digital_scores) > 0 and len(traditional_scores) > 0:
        t_stat, p_value = stats.ttest_rel(digital_scores, traditional_scores)
        print(f"対応のあるt検定 (Paired t-test): t={t_stat:.2f}, p={p_value:.3f}")
        if p_value < 0.05:
            print("→ 統計的有意差あり (Statistically significant difference)")
        else:
            print("→ 統計的有意差なし (No statistically significant difference)")
    print()
    
    print("=== 相関分析 (Correlation Analysis) ===")
    if tech_col in df.columns and digital_sat_col in df.columns:
        tech_digital_corr = df[tech_col].corr(df[digital_sat_col])
        if not pd.isna(tech_digital_corr):
            print(f"テクノロジー親和性 vs デジタル満足度: {tech_digital_corr:.3f}")
    
    if digital_pref_col in df.columns and digital_sat_col in df.columns:
        pref_digital_corr = df[digital_pref_col].corr(df[digital_sat_col])
        if not pd.isna(pref_digital_corr):
            print(f"デジタル嗜好 vs デジタル満足度: {pref_digital_corr:.3f}")
    
    print("\n=== AR/VR比較統計的検定結果 (AR/VR Comparison Statistical Test Results) ===")
    ar_vr_aspects = {
        '[a]見た目がリアル': '[a]見た目がリアル',
        '[b]体験が楽しい': '[b]体験が楽しい', 
        '[c]操作がしやすい': '[c]操作がしやすい',
        '[d]手軽だった': '[d]手軽だった'
    }
    
    for aspect_name, col_name in ar_vr_aspects.items():
        if col_name in df.columns:
            data = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if len(data) > 0:
                t_stat, p_value = stats.ttest_1samp(data, 3.0)
                cohens_d = (data.mean() - 3.0) / data.std() if data.std() > 0 else 0
                
                if p_value < 0.001:
                    sig_text = "***"
                elif p_value < 0.01:
                    sig_text = "**"
                elif p_value < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "n.s."
                
                print(f"{aspect_name}: 平均={data.mean():.2f}, t={t_stat:.2f}, p={p_value:.3f} {sig_text}, d={cohens_d:.2f}")
    
    print("\n=== デバッグ情報 (Debug Info) ===")
    print("利用可能な列 (Available columns):")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nデータ形状 (Data shape): {df.shape}")

def main():
    df = load_and_clean_data()
    
    create_demographics_plot(df)
    print("✓ 人口統計分析グラフを生成しました (Demographics analysis created)")
    
    create_satisfaction_comparison(df)
    print("✓ 満足度比較グラフを生成しました (Satisfaction comparison created)")
    
    create_technology_correlation(df)
    print("✓ テクノロジー相関分析グラフを生成しました (Technology correlation created)")
    
    create_comparison_radar(df)
    print("✓ 分布付き箱ヒゲ図比較を生成しました (Box plot with distribution comparison created)")
    
    print_summary_statistics(df)

if __name__ == "__main__":
    main()
