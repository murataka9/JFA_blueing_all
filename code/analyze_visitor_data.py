import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager as fm

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Takao Gothic', 'IPAex Gothic', 'DejaVu Sans']

def load_and_clean_data():
    df = pd.read_csv('Visitors/visitors survey_data.csv')
    
    numeric_columns = ['年齢 (Age)', 'サッカーへの関心 (Interest in Soccer)', 
                      'テクノロジーへの親和性 (Technology Affinity)', 
                      'デジタル展示への嗜好 (Digital Exhibition Preference)',
                      'デジタル展示満足度 (Digital Exhibition Satisfaction)',
                      '実物展示満足度 (Traditional Exhibition Satisfaction)']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    digital_col = 'デジタル展示満足度 (Digital Exhibition Satisfaction)'
    traditional_col = '実物展示満足度 (Traditional Exhibition Satisfaction)'
    
    digital_scores = df[digital_col].dropna()
    traditional_scores = df[traditional_col].dropna()
    
    categories = ['満足度\n(Satisfaction)', '平均評価\n(Average Rating)', 
                 '最高評価\n(Max Rating)', '評価の一貫性\n(Consistency)']
    
    digital_values = [
        digital_scores.mean(),
        digital_scores.mean(),
        digital_scores.max(),
        5 - digital_scores.std()
    ]
    
    traditional_values = [
        traditional_scores.mean(),
        traditional_scores.mean(),
        traditional_scores.max(),
        5 - traditional_scores.std()
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    digital_values += digital_values[:1]
    traditional_values += traditional_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, digital_values, 'o-', linewidth=2, label='デジタル展示 (Digital)', color='#FF6B6B')
    ax.fill(angles, digital_values, alpha=0.25, color='#FF6B6B')
    
    ax.plot(angles, traditional_values, 'o-', linewidth=2, label='実物展示 (Traditional)', color='#4ECDC4')
    ax.fill(angles, traditional_values, alpha=0.25, color='#4ECDC4')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 5)
    ax.set_title('デジタル展示 vs 実物展示の比較\n(Digital vs Traditional Exhibition Comparison)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('Visitors/comparison_radar.png', dpi=150, bbox_inches='tight')
    plt.close()

def print_summary_statistics(df):
    print("=== 来場者調査データ分析結果 (Visitor Survey Analysis Results) ===\n")
    
    age_col = '年齢 (Age)'
    soccer_col = 'サッカーへの関心 (Interest in Soccer)'
    tech_col = 'テクノロジーへの親和性 (Technology Affinity)'
    digital_pref_col = 'デジタル展示への嗜好 (Digital Exhibition Preference)'
    digital_sat_col = 'デジタル展示満足度 (Digital Exhibition Satisfaction)'
    traditional_sat_col = '実物展示満足度 (Traditional Exhibition Satisfaction)'
    
    print(f"回答者数 (Total Respondents): {len(df)}")
    print(f"平均年齢 (Average Age): {df[age_col].mean():.1f}歳 ({df[age_col].min():.0f}-{df[age_col].max():.0f}歳)")
    print(f"サッカー関心度 (Soccer Interest): {df[soccer_col].mean():.1f}/5")
    print(f"テクノロジー親和性 (Technology Affinity): {df[tech_col].mean():.1f}/5")
    print(f"デジタル展示嗜好 (Digital Exhibition Preference): {df[digital_pref_col].mean():.1f}/5")
    print()
    
    digital_scores = df[digital_sat_col].dropna()
    traditional_scores = df[traditional_sat_col].dropna()
    
    print("=== 満足度比較 (Satisfaction Comparison) ===")
    print(f"デジタル展示満足度 (Digital): {digital_scores.mean():.1f}/5 ({len(digital_scores)}名)")
    print(f"実物展示満足度 (Traditional): {traditional_scores.mean():.1f}/5 ({len(traditional_scores)}名)")
    print()
    
    tech_digital_corr = df[tech_col].corr(df[digital_sat_col])
    pref_digital_corr = df[digital_pref_col].corr(df[digital_sat_col])
    
    print("=== 相関分析 (Correlation Analysis) ===")
    print(f"テクノロジー親和性 vs デジタル満足度: {tech_digital_corr:.3f}")
    print(f"デジタル嗜好 vs デジタル満足度: {pref_digital_corr:.3f}")

def main():
    df = load_and_clean_data()
    
    create_demographics_plot(df)
    print("✓ 人口統計分析グラフを生成しました (Demographics analysis created)")
    
    create_satisfaction_comparison(df)
    print("✓ 満足度比較グラフを生成しました (Satisfaction comparison created)")
    
    create_technology_correlation(df)
    print("✓ テクノロジー相関分析グラフを生成しました (Technology correlation created)")
    
    create_comparison_radar(df)
    print("✓ レーダーチャート比較を生成しました (Radar chart comparison created)")
    
    print_summary_statistics(df)

if __name__ == "__main__":
    main()
