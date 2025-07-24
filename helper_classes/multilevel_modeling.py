# %%
import os # type:ignore # isort:skip # fmt:skip # noqa # nopep8
import sys # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path

from pandas import DataFrame # type:ignore # isort:skip # fmt:skip # noqa # nopep8

mod = sys.modules[__name__]

code_dir = None
code_dir_name = 'Code'
unwanted_subdir_name = 'Analysis'

if code_dir_name not in str(Path.cwd()).split('/')[-1]:
    for _ in range(5):

        parent_path = str(Path.cwd().parents[_]).split('/')[-1]

        if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):

            code_dir = str(Path.cwd().parents[_])

            if code_dir is not None:
                break
else:
    code_dir = str(Path.cwd())
sys.path.append(code_dir)

# %load_ext autoreload
# %autoreload 2


# %%
from setup_module.imports import *  # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from setup_module import researchpy_fork as rp # type:ignore # isort:skip # fmt:skip # noqa # nopep8

# %%
class MultiLevelModeling:

    def __init__(self, hypothesis_num: str, df: pd.DataFrame, dv: str, ivs_list: list[str], cols_cov_list: list[str] = None, levels_list: list[str] = None, moderators_list: list[str] = None, ols_only: bool = None, iv_interactions: bool = None, df_bisect_col: str = None, df_bisect_cat: str = None, remove_biset_col: bool = None, switch_enabled: bool = None, decimals: int = None, print_summary: bool = None, plot_enabled: bool = None, model_compare_enabled: bool = None, save_enabled: bool = None, model_save_enabled: bool = None, sm_check: bool = None, fitted_models: dict = None):
        self.hypothesis_num: str = hypothesis_num
        self.df_bisect_col = None if not df_bisect_col else df_bisect_col
        self.df_bisect_cat = None if not df_bisect_cat else df_bisect_cat
        self.remove_biset_col = remove_biset_col if remove_biset_col is not None else False
        self.df: pd.DataFrame = df if self.df_bisect_col is None else (
            df.loc[df[self.df_bisect_col] != self.df_bisect_cat] if self.remove_biset_col
            else df.loc[df[self.df_bisect_col] == self.df_bisect_cat]
        )
        self.dv: str = dv
        self.ivs_list: list = ivs_list
        self.cols_cov_list: list = cols_cov_list if cols_cov_list is not None else []
        self.levels_list: list = levels_list if levels_list is not None else (['participant_id', 'task'] if self.df['participant_id'].duplicated().sum() == 0 else [])
        self.moderators_list: list = moderators_list if moderators_list is not None else []
        self.ols_only: bool = ols_only if ols_only is not None else (True if len(self.levels_list) == 0 else False)
        self.iv_interactions: bool = iv_interactions if iv_interactions is not None else True
        self.switch_enabled: bool = switch_enabled if switch_enabled is not None else False
        self.decimals: int = decimals if decimals is not None else 2
        self.print_summary: bool = print_summary if print_summary is not None else True
        self.plot_enabled: bool = plot_enabled if plot_enabled is not None else True
        self.model_compare_enabled: bool = model_compare_enabled if model_compare_enabled is not None else True
        self.save_enabled: bool = save_enabled if save_enabled is not None else True
        self.model_save_enabled: bool = model_save_enabled if model_save_enabled is not None else False
        self.sm_check: bool = sm_check if sm_check is not None else True
        self.fitted_models = fitted_models if fitted_models is not None else defaultdict(lambda: defaultdict(dict))

    def _get_color_mapping(self, categories: list[str]) -> list[str]:
        gray = colorblind_hex_colors[2]
        remaining = [c for j,c in enumerate(colorblind_hex_colors) if j != 2]
        mapping = {}
        rem_idx = 0
        for cat in categories:
            cat_str = str(cat)
            if cat_str.lower() in ('no_recommendation', 'no recommendation'):
                mapping[cat] = gray
            else:
                mapping[cat] = remaining[rem_idx % len(remaining)]
                rem_idx += 1
        return mapping

    @property
    def cond(self):
        return lambda col: self.df[col].dtype.name == 'category' or '_dummy' in col or '_num' not in col

    def order_cat_cols(self):

        for col in self.ivs_list + self.cols_cov_list:
            if self.cond(col):
                self.df[col] = self.df[col].astype('category')

                if self.switch_enabled:
                    if 'female' in self.df[col].cat.categories:
                        self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['female', 'male'])
                    elif 'warmth' in self.df[col].cat.categories:
                        self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['warmth', 'competence'])
                else:
                    if 'female' in self.df[col].cat.categories:
                        if 'no_recommendation' in self.df[col].cat.categories:
                            self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['male', 'no_recommendation', 'female'])
                        else:
                            self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['male', 'female'])

                    elif 'warmth' in self.df[col].cat.categories:
                        self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['competence', 'warmth'])

                    elif 'stereotype_fit' in self.df[col].cat.categories:
                        self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['qualifications_fit', 'stereotype_fit'])

                    elif 'match' in self.df[col].cat.categories:
                        self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['mismatch', 'match'])

                    elif 'recommended' in self.df[col].cat.categories:
                        self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['not_recommended', 'no_recommendation', 'recommended'])

                    elif 'bias_recommendation' in self.df[col].cat.categories:
                        self.df[col] = pd.Categorical(self.df[col], ordered=False, categories=['unbias_recommendation', 'no_recommendation', 'bias_recommendation'])

    def print_var_info(self):
        print('='*20)
        print(f'Variable information for {self.hypothesis_num.capitalize()} {rename_predictor_name(self.dv)}')
        print('='*20)
        print(f'\tDataframe shape: {self.df.shape}')
        print(f'\tDependent variable: {self.dv}')
        print(f'\tIndependent variables: {self.ivs_list}')
        print(f'\tModerators: {self.moderators_list}')
        print(f'\tCovariates: {self.cols_cov_list}')
        print(f'\tLevels: {self.levels_list}')
        if self.df_bisect_col is not None:
            print(f'\tBisecting df on column: {self.df_bisect_col} to keep {self.df_bisect_cat}')
            print(f'Value counts for {self.df_bisect_col}:')
            pprint.pprint(self.df[self.df_bisect_col].value_counts())
        print(f'\tIV interactions enabled: {self.iv_interactions}')
        print(f'\tPrint summary: {self.print_summary}')
        print(f'\tSM check: {self.sm_check}')
        print(f'\tPerforming OLS regression only: {self.ols_only}')
        print(f'\tSwitch enabled: {self.switch_enabled}')
        print(f'\tModel comparison enabled: {self.model_compare_enabled}')
        print(f'\tPlotting enabled: {self.plot_enabled}')
        print(f'\tSaving enabled: {self.save_enabled}')
        print(f'\tModel saving enabled: {self.model_save_enabled}')
        print('\n')

    def print_means_info(self):
        dv_str = rename_predictor_name(self.dv)
        print('='*20)
        print(f'Means for DV: {dv_str}')
        print('='*20)
        pprint.pprint(rp.summary_cont(self.df[self.dv]).round(2))
        print('\n\n')

        cat_col = [col for col in self.ivs_list + self.moderators_list if self.cond(col)]

        for iv in cat_col:
            print('='*20)
            print(f'Counts for IV: {rename_predictor_name(iv)}:')
            print('='*20)
            pprint.pprint(rp.summary_cat(self.df[iv]))
            print('\n\n')

            for cat in self.df[iv].cat.categories:
                try:
                    means_summary = rp.summary_cont(self.df[f'{iv}_{cat}']).round(2)
                    means_summary_str = f'{iv}_{cat}'
                    means_on_dv_summary = rp.summary_cont(self.df[self.df[iv] == cat][self.dv]).round(2)
                except:
                    means_summary_str = rename_predictor_name(f'{iv}_num')
                    means_summary = rp.summary_cont(self.df[f'{iv}_num']).round(2)

                print('='*20)
                print(f'Means for IV: {means_summary_str}:')
                print('='*20)
                pprint.pprint(means_summary)
                print('\n\n')
                try:
                    print('='*20)
                    print(f'Means for IV: {means_summary_str} on DV {dv_str}:')
                    print('='*20)
                    pprint.pprint(means_on_dv_summary)
                    print('\n\n')
                except:
                    pass

        if len(cat_col) > 1:
            for iv1, iv2 in itertools.combinations(cat_col, 2):
                var_str = rename_predictor_name(f'{iv1} x {iv2}')
                print('='*20)
                print(f'Means for {self.hypothesis_num.capitalize()}: {dv_str} on {var_str}')
                print('='*20)
                pprint.pprint(rp.summary_cont(self.df.groupby([iv1, iv2])[self.dv]).round(2))
                print('\n\n')

    def print_ttest_info(self):

        # Identify categorical columns
        cat_cols = [col for col in self.ivs_list + self.moderators_list if self.cond(col)]
        dv_str = rename_predictor_name(self.dv)

        tt_args = defaultdict(dict)
        levene_args = defaultdict(dict)

        if not cat_cols:
            print('No categorical predictors found.')
            return

        for col in cat_cols:
            cats = self.df[col].astype('category').cat.categories.astype(str).tolist()
            valid_cats = [
                cat for cat in cats if self.df.loc[self.df[col].astype(str) == cat, self.dv].notna().all()
            ]

            if len(valid_cats) < 2:
                print(f'Skipping {col}: fewer than two valid categories (found {len(valid_cats)})')
                continue

            # Pairwise comparisons
            for iv1, iv2 in itertools.combinations(valid_cats, 2):
                var_str = rename_predictor_name(f'{col}_{iv1} x {col}_{iv2}')
                # Mask and drop NaNs
                mask1 = (self.df[col].astype(str) == iv1) & self.df[self.dv].notna()
                mask2 = (self.df[col].astype(str) == iv2) & self.df[self.dv].notna()

                x = self.df.loc[mask1, self.dv]
                y = self.df.loc[mask2, self.dv]

                if len(x) < 2 or len(y) < 2:
                    print(f'Skipping t-test for {col}: {iv1} (n={len(x)}), {iv2} (n={len(y)})')
                    continue

                tt_args[var_str] = {
                    'x': x,
                    'y': y,
                }
                levene_args[var_str] = {
                    'data': self.df,
                    'dv': self.dv,
                    'group': col,
                }

            for var_str, args in tt_args.items():
                x = args['x']
                y = args['y']
                print(f'{var_str}: length x = {len(x)}, length y = {len(y)}')
                levens_result = pg.homoscedasticity(
                    **levene_args[var_str],
                    method='levene',
                )
                ttest_result = pg.ttest(
                    **tt_args[var_str],
                    paired=False,
                )
                title = f'T-test for {self.hypothesis_num.capitalize()}: {dv_str} on {var_str}'
                if self.df_bisect_col is not None:
                    title += f' - Only for {rename_predictor_name(self.df_bisect_col)} = {rename_predictor_name(self.df_bisect_cat)}'
                print('='*20)
                print(title)
                print('='*20)
                pprint.pprint(levens_result)
                print('\n')
                print('-'*20)
                pprint.pprint(ttest_result)
                print('\n\n')
        else:
            print(f'Cannot print t-test for {self.hypothesis_num.capitalize()} {dv_str} because not all variables are categorical or some categories have all missing values.')
            print('\n\n')

    def print_descriptives_info(self):
        dv_str = rename_predictor_name(self.dv)
        self.print_means_info()
        if all(
            self.cond(col)
            for col in self.ivs_list
        ):
            self.print_ttest_info()
        else:
            print(f'Cannot print ttest for {self.hypothesis_num.capitalize()}: {dv_str} because not all variables are categorical.')
            print('\n\n')

    def plot_pointplot(self):
        close_plots(plt)
        dv_str = rename_predictor_name(self.dv)
        if all(
            self.cond(col)
            for col in self.ivs_list
        ):
            pointplot_args = defaultdict(dict)
            if len(self.ivs_list) > 1:
                for iv1, iv2 in itertools.combinations(self.ivs_list, 2):
                    var_str = rename_predictor_name(f'{iv1} x {iv2}')
                    x_label_str = rename_predictor_name(iv2)
                    pointplot_args[var_str] = {
                        'x': iv2,
                        'hue': iv1,
                    }
            else:
                var_str = rename_predictor_name(self.ivs_list[0])
                x_label_str = rename_predictor_name(self.ivs_list[0])
                pointplot_args[var_str] = {
                    'x': self.ivs_list[0],
                }

            for var_str, args in pointplot_args.items():
                title = f'Pointplot for {self.hypothesis_num.capitalize()} {dv_str} on {var_str}'
                if self.df_bisect_col is not None:
                    title += f' - Only for {rename_predictor_name(self.df_bisect_col)} = {rename_predictor_name(self.df_bisect_cat)}'
                print('='*20)
                print(title)
                print('='*20)
                plt.figure(figsize=(6, 6))
                if 'hue' in args:
                    hue_var = args['hue']
                    # force a consistent order
                    hue_levels = (
                        self.df[hue_var].cat.categories.tolist()
                        if self.df[hue_var].dtype.name == 'category'
                        else list(self.df[hue_var].unique())
                    )
                    palette = self._get_color_mapping(hue_levels)
                    ax = sns.pointplot(
                        data=self.df,
                        y=self.dv,
                        hue=hue_var,
                        hue_order=hue_levels,
                        palette=palette,
                        **{k:v for k,v in args.items() if k!='hue'},
                    )
                else:
                    palette = self._get_color_mapping(self.df[args['x']].cat.categories.tolist())
                    ax = sns.pointplot(data=self.df, y=self.dv, palette=palette, **args)

                # Update x-tick labels using rename_predictor_name
                xticks = ax.get_xticks()
                xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
                ax.set_xticklabels([rename_predictor_name(label) for label in xtick_labels])

                # Update legend labels using rename_predictor_name
                if len(self.ivs_list) > 1:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, [rename_predictor_name(label) for label in labels], title=rename_predictor_name(var_str))

                plt.title(title, fontsize=14)
                plt.xlabel(x_label_str)
                plt.ylabel(dv_str)
                plt.grid(True)
                plt.tight_layout()

                if self.save_enabled:
                    img_save_path = f'{hypotheses_plot_save_path}{self.hypothesis_num}_{self.dv}_{var_str}_pointplot'
                    if self.df_bisect_col is not None:
                        img_save_path += f'_{self.df_bisect_cat}'
                    for image_save_format in ['png']:
                        plt.savefig(
                            f'{img_save_path}.{image_save_format}',
                            format=image_save_format, dpi=3000, bbox_inches='tight',
                        )
                show_and_close_plots(plt)
                print('\n\n')
        else:
            print(f'Cannot plot pointplot for {self.hypothesis_num.capitalize()} {dv_str} because not all variables are categorical.')
            print('\n\n')

    def plot_interactionplot(self):
        close_plots(plt)
        dv_str = rename_predictor_name(self.dv)
        trace_label_str = rename_predictor_name(self.ivs_list[0])

        if len(self.ivs_list) > 1 and all(
            self.cond(col)
            for col in self.ivs_list
        ):
            for iv1, iv2 in itertools.combinations(self.ivs_list, 2):
                var_str = rename_predictor_name(f'{iv1} x {iv2}')
                trace_label_str = rename_predictor_name(iv1)
                x_label_str = rename_predictor_name(iv2)
                cat_main_label_str = rename_predictor_name(self.df[iv2].cat.categories[-1])
                cat_other_label_str = rename_predictor_name(self.df[iv2].cat.categories[-2])
                title = f'Interaction plot for {self.hypothesis_num.capitalize()} {dv_str} on {trace_label_str} x {x_label_str}'
                print('='*20)
                print(title)
                print('='*20)
                self.markers = ['D', '^', 'o']
                self.colors = ['C0', 'C1', 'C5']
                fig, ax = plt.subplots(figsize=(6, 6))
                # Determine the trace categories in order
                trace_levels = (
                    self.df[iv1].cat.categories.tolist()
                    if self.df[iv1].dtype.name == 'category'
                    else list(self.df[iv1].unique())
                )
                # Build explicit category→color mapping
                color_mapping = self._get_color_mapping(trace_levels)
                # Extract colors in the category order
                colors = [color_mapping[cat] for cat in trace_levels]
                interaction_plot(
                    x=self.df[f'{iv2}_{self.df[iv2].cat.categories[-1]}'],
                    response=self.df[self.dv],
                    trace=self.df[iv1],
                    markers=[self.markers[i] for i in range(len(self.df[iv1].unique()))],
                    colors=colors,
                    ms=10,
                    ax=ax,
                    xlabel=x_label_str,
                    ylabel=dv_str,
                    legendtitle=trace_label_str,
                )

                # Update x-tick labels using rename_predictor_name
                ax.set_xticks([0, 1])
                ax.set_xticklabels(
                    [
                        rename_predictor_name(self.df[iv2].cat.categories[0]),
                        rename_predictor_name(self.df[iv2].cat.categories[-1]),
                    ],
                )

                # Rename x-tick labels
                xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
                ax.set_xticklabels([rename_predictor_name(label) for label in xtick_labels])

                # Rename legend labels
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [rename_predictor_name(label) for label in labels]
                ax.legend(handles, new_labels, title=trace_label_str)

                plt.title(title, fontsize=14)
                plt.grid(True)
                plt.tight_layout()

                if self.save_enabled:
                    img_save_path = f'{hypotheses_plot_save_path}{self.hypothesis_num}_{self.dv}_{var_str}_interactionplot'
                    if self.df_bisect_col is not None:
                        img_save_path += f'_{self.df_bisect_cat}'
                    for image_save_format in ['png']:
                        plt.savefig(
                            f'{img_save_path}.{image_save_format}',
                            format=image_save_format, dpi=3000, bbox_inches='tight',
                        )
                show_and_close_plots(plt)
                print('\n\n')

        else:
            print(f'Cannot plot interaction plot for {self.hypothesis_num.capitalize()} {dv_str} on {trace_label_str} because there is only one independent variable.')
            print('\n\n')

    def plot_moderation_interactionplot(self):
        close_plots(plt)
        if len(self.moderators_list) > 0 and all(self.cond(col) for col in self.ivs_list):
            moderation_interactionplot_args = defaultdict(dict)
            dv_str = rename_predictor_name(self.dv)
            for mod, iv in tqdm_product(self.moderators_list, self.ivs_list):
                mod_str = rename_predictor_name(mod)
                var_str = rename_predictor_name(f'{mod} x {iv}')
                moderation_interactionplot_args[var_str] = {
                    'x': mod,
                    'y': self.dv,
                    'hue': iv,
                }

            for var_str, args in moderation_interactionplot_args.items():

                title = f'Moderation interaction plot for {self.hypothesis_num.capitalize()} {dv_str} on {var_str}'
                if self.df_bisect_col is not None:
                    title += f' - Only for {rename_predictor_name(self.df_bisect_col)} = {rename_predictor_name(self.df_bisect_cat)}'
                print('='*20)
                print(title)
                print('='*20)
                hue_var = args['hue']
                hue_levels = (
                    self.df[hue_var].cat.categories.tolist()
                    if self.df[hue_var].dtype.name == 'category'
                    else list(self.df[hue_var].unique())
                )
                palette = self._get_color_mapping(hue_levels)
                fig = sns.lmplot(
                    data=self.df,
                    height=6,
                    aspect=1.5,
                    ci=95,
                    scatter=False,
                    palette=palette,
                    **moderation_interactionplot_args[var_str],
                )
                fig._legend.remove()
                ax = fig.ax
                old_title = fig._legend.get_title().get_text()
                new_title = rename_predictor_name(old_title)
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [rename_predictor_name(lab) for lab in labels]
                ax.legend(handles, new_labels, title=new_title)
                plt.xlabel(mod_str)
                plt.ylabel(dv_str)
                plt.title(title, fontsize=14)
                plt.grid(True)
                fig.tight_layout()
                if self.save_enabled:
                    img_save_path = f'{hypotheses_plot_save_path}{self.hypothesis_num}_{self.dv}_{self.moderators_list[0]}_moderation_interactionplot'
                    if self.df_bisect_col is not None:
                        img_save_path += f'_{self.df_bisect_cat}'
                    for image_save_format in ['png']:
                        plt.savefig(
                            f'{img_save_path}.{image_save_format}',
                            format=image_save_format, dpi=3000, bbox_inches='tight',
                        )
                show_and_close_plots(plt)
                print('\n\n')

        else:
            print(f'Cannot plot moderation interaction plot for {self.hypothesis_num.capitalize()} {rename_predictor_name(self.dv)} because there are no moderators.')
            print('\n\n')

    def descriptives(self):
        if self.print_summary:
            dv_str = rename_predictor_name(self.dv)
            if len(self.ivs_list) > 1:
                var_str = ' x '.join([rename_predictor_name(iv) for iv in self.ivs_list])
            else:
                var_str = rename_predictor_name(self.ivs_list[0])
            print('='*20)
            print(f'Descriptives for {self.hypothesis_num.capitalize()} DV: {dv_str} on {var_str}')
            print('='*20)
            self.print_var_info()
            self.print_descriptives_info()
            if self.plot_enabled:
                self.plot_pointplot()
                self.plot_interactionplot()
                self.plot_moderation_interactionplot()

    def make_reference_cat_dict(self):
        self.reference_cat_dict = {}

        for col in self.ivs_list + self.cols_cov_list:
            if self.cond(col):
                self.reference_cat_dict[col] = [
                    cat for cat in self.df[col].astype('category').cat.categories.astype(str)
                    if not self.df.loc[self.df[col] == cat].isna().all().all()
                ]
        return self.reference_cat_dict

    def make_sm_cat(self, col: str, with_cat: bool) -> str:
        cats = [
            cat for cat in self.df[col].astype('category').cat.categories.astype(str)
            if not self.df.loc[self.df[col] == cat].isna().all().all()
        ]
        formula_with_cats = []
        for cat in cats:
            if len(self.reference_cat_dict[col]) != 0:
                reference_cat = self.reference_cat_dict[col][0] if isinstance(self.reference_cat_dict[col], list) else self.reference_cat_dict[col]
                if cat != reference_cat:
                    if with_cat:
                        formula_with_cats.append(f'C({col}, Treatment("{reference_cat}"))[T.{cat}]')
                    else:
                        formula_with_cats.append(f'C({col})[T.{cat}]')
        return formula_with_cats

    def make_var_str(self, lst: list[str], with_cat: bool) -> list[str]:
        self.temp_list = []
        for col in lst:
            if self.cond(col) and with_cat:
                if not self.ols_only:
                    self.temp_list.append(f'C({col}){self.reference_cat_dict[col][0]}')
                elif self.ols_only:
                    self.temp_list.extend(self.make_sm_cat(col, with_cat=with_cat))
            elif self.cond(col) and not with_cat:
                self.temp_list.append(f'C({col})')
            elif not self.cond(col):
                self.temp_list.append(col)
        return self.temp_list

    def make_formula_list(self):
        print(f'Making formula list for {self.hypothesis_num.capitalize()} {rename_predictor_name(self.dv)} with {len(self.ivs_list)} IVs and {len(self.moderators_list)} moderators')

        iv_strs = self.make_var_str(self.ivs_list, with_cat=False)
        mod_strs = self.make_var_str(self.moderators_list, with_cat=False)

        if self.ivs_list:
            self.formula_list = [
                # Model 0
                f'{self.dv} ~ 1',
            ]

        base = ''
        for i, iv in enumerate(iv_strs):
            if i == 0:
                base = f'{self.dv} ~ {iv}'
            else:
                base = base + f' + {iv}'
            self.formula_list.append(base)

            if self.iv_interactions:
                for prev_iv in iv_strs[:i]:
                    self.formula_list.append(base + f' + {prev_iv} * {iv}')

        # Stepwise add each moderator’s main effect, then interactions for each IV
        if mod_strs:
            # 1) Append each moderator main effect in sequence
            for mod in mod_strs:
                last = self.formula_list[-1]
                self.formula_list.append(last + f' + {mod}')
            # 2) For each IV, append interactions with all moderators in sequence
            for iv in iv_strs:
                for mod in mod_strs:
                    last = self.formula_list[-1]
                    self.formula_list.append(last + f' + {iv} * {mod}')
            if self.iv_interactions:
                # 3) All interactions: any combination of IVs (size ≥2) with any combination of moderators (size ≥1)
                for r in range(2, len(iv_strs) + 1):
                    for iv_group in itertools.combinations(iv_strs, r):
                        for s in range(1, len(mod_strs) + 1):
                            for mod_group in itertools.combinations(mod_strs, s):
                                term = ' * '.join(iv_group + mod_group)
                                last = self.formula_list[-1]
                                self.formula_list.append(last + f' + {term}')
        return self.formula_list

    def make_sm_formula_list(self):
        def repl(match):
            col = match.group(1)

            if self.cond(col):
                return f'C({col}, Treatment("{self.reference_cat_dict[col][0]}"))'
            else:
                return match.group(0)

        pattern = r'C\((\w+)\)'
        sm_formula_list = []
        self.formula_list = self.make_formula_list()
        for formula in self.formula_list:
            new_formula = re.sub(pattern, repl, formula)
            sm_formula_list.append(new_formula)
        return sm_formula_list

    def make_covariate_formula(self):
        if len(self.cols_cov_list) != 0:
            return f' + '.join(self.make_var_str(self.cols_cov_list, with_cat=False))
        else:
            return None

    def make_level_formula(self):
        if len(self.levels_list) == 1:
            self.level_formula = f'(1 | {self.levels_list[0]})'
        elif len(self.levels_list) > 1:
            levels_str = '/'.join(level for level in self.levels_list)
            self.level_formula = f'(1 | {levels_str})'
        else:
            self.level_formula = None
        return self.level_formula

    def plot_model(self, model):
        close_plots(plt)
        print('='*20)
        print(f'Plotting results for model {self.hypothesis_num}:\n{model.formula}')
        print('='*20)
        model.plot_summary(figsize=(16, 6), plot_intercept=False)
        plt.xlabel(xlabel=rename_predictor_name(self.dv))
        plt.ylabel('Predictor(s)')

        ax = plt.gca()

        old_predictors = [col for col in model.coefs.index.tolist() if 'Intercept' not in col]
        new_predictors = [rename_predictor_name(p) for p in old_predictors]
        ax.set_yticklabels(new_predictors)

        plt.title(f'Multilevel Model {self.hypothesis_num.capitalize()} for {rename_predictor_name(self.dv)} on {" x ".join([rename_predictor_name(col) for col in self.ivs_list])}', fontsize=16)
        plt.tight_layout()
        plt.tick_params(labelsize=14)
        plt.tight_layout()

        if self.save_enabled:
            img_save_path = f'{hypotheses_plot_save_path}{self.hypothesis_num}_{self.dv}_estimatesplot'
            if self.df_bisect_col is not None:
                img_save_path += f'_{self.df_bisect_cat}'
            for image_save_format in ['png']:
                plt.savefig(
                    f'{img_save_path}.{image_save_format}',
                    format=image_save_format, dpi=3000, bbox_inches='tight',
                )
        show_and_close_plots(plt)
        print('\n')

    def plot_sm_model(self, sm_model):
        close_plots(plt)
        figsize = (16, 6)
        params = sm_model.params
        conf_int = sm_model.conf_int()

        exog_names = [name for name in sm_model.model.exog_names if name not in ['const', 'Intercept']]
        ordered_params = params[exog_names]
        ordered_conf_int = conf_int.loc[exog_names]

        lower = ordered_conf_int.iloc[:, 0]
        upper = ordered_conf_int.iloc[:, 1]

        error_lower = ordered_params - lower
        error_upper = upper - ordered_params

        n_params = len(ordered_params)
        if figsize[1] is None:
            height = max(2, n_params * 0.5)
            figsize = (figsize[0], height)

        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.01, right=0.95)

        positions = np.arange(n_params)

        ax.errorbar(
            ordered_params, positions,
            xerr=[error_lower, error_upper],
            fmt='o', color='black', capsize=5, markersize=5,
        )
        padding = 0.5
        ax.set_ylim(positions[0] - padding, positions[-1] + padding)
        ax.margins(y=0.2)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(positions)
        old_predictors = [col for col in ordered_params.index.tolist() if 'Intercept' not in col]
        new_predictors = [rename_predictor_name(p) for p in old_predictors]
        ax.set_yticklabels(new_predictors)
        ax.tick_params(axis='both', labelsize=12)
        ax.invert_yaxis()
        ax.set_xlabel(rename_predictor_name(self.dv), fontsize=14)
        ax.set_ylabel('Predictor(s)', fontsize=14)
        ax.set_title(f'OLS Model {self.hypothesis_num.capitalize()} for {rename_predictor_name(self.dv)} on {" x ".join([rename_predictor_name(col) for col in self.ivs_list])}', fontsize=16)
        plt.tight_layout()

        if self.save_enabled:
            image_save_path = f'{hypotheses_plot_save_path}{self.hypothesis_num}_{self.dv}_sm_estimatesplot'
            if self.df_bisect_col is not None:
                image_save_path += f'_{self.df_bisect_cat}'
            for image_save_format in ['png']:
                plt.savefig(
                    f'{image_save_path}.{image_save_format}',
                    format=image_save_format, dpi=3000, bbox_inches='tight',
                )
        show_and_close_plots(plt)
        print('\n')

    def fit_stepwise_models(self):
        self.formula_list = self.make_formula_list()
        self.level_formula = self.make_level_formula()
        self.cov_formula = self.make_covariate_formula()

        self.model_results_list = []

        for i, formula in tqdm.tqdm(enumerate(self.formula_list)):
            if self.cov_formula is None or ('1' in formula and i == 0):
                full_formula = f'{formula} + {self.level_formula}'
            else:
                full_formula = f'{formula} + {self.cov_formula} + {self.level_formula}'
            model = pymer4.models.Lmer(full_formula, data=self.df)
            if self.reference_cat_dict is not None:
                model.fit(factors=self.reference_cat_dict, REML=True, summarize=False)
            else:
                model.fit(REML=True, summarize=False)

            if self.model_save_enabled:
                pymer4_model_save_path = f'{hypotheses_model_save_path}{self.hypothesis_num}_{self.dv}_model{i}'
                if self.df_bisect_col is not None:
                    pymer4_model_save_path += f'_{self.df_bisect_cat}'
                pymer4.io.save_model(model, f'{pymer4_model_save_path}.joblib')

            self.model_results_list.append(model)

            if self.print_summary:
                print('\n')
                print('='*20)
                print(f'MODEL {i} for {self.hypothesis_num.capitalize()}:')
                if self.df_bisect_col is not None:
                    print(f'\tBisecting df on column: {self.df_bisect_col} to keep {self.df_bisect_cat}')
                print(f'\tRunning multilevel model: {formula}')
                print('='*20)
                print('\n')
                print('='*20)
                print(f'Reference categories dict for {self.hypothesis_num.capitalize()} {self.dv}:')
                print('='*20)
                pprint.pprint(self.reference_cat_dict)
                print('\n')
                print('='*20)
                pprint.pprint(model.summary())
                print('\n')
                print('='*20)
                print(f'ANOVAs model {i} for {self.hypothesis_num.capitalize()}:')
                print('='*20)
                pprint.pprint(model.anova())
                print('\n\n')
                if self.plot_enabled and i == len(self.formula_list) - 1:
                    self.plot_model(model)
                    print('\n\n')

        return self.model_results_list

    def fit_stepwise_sm_models(self):
        self.sm_formula_list = self.make_sm_formula_list()
        self.sm_cov_formula = self.make_covariate_formula()

        self.sm_model_results_list = []

        for i, formula in tqdm.tqdm(enumerate(self.sm_formula_list)):
            if self.sm_cov_formula is None or ('1' in formula and i == 0):
                full_formula = formula
            else:
                full_formula = f'{formula} + {self.sm_cov_formula}'

            if not self.ols_only:
                sm_model = smf.mixedlm(full_formula, data=self.df, groups=self.df[self.levels_list[0]]).fit(disp=False)
            elif self.ols_only:
                sm_model = smf.ols(full_formula, data=self.df).fit(disp=False);
            if self.model_save_enabled:
                sm_model.save(f'{hypotheses_model_save_path}{self.hypothesis_num}_{self.dv}_sm_model{i}.joblib')

            self.sm_model_results_list.append(sm_model)

            if self.print_summary:
                print('\n')
                print('='*20)
                print(f'SM MODEL {i} for {self.hypothesis_num.capitalize()}:')
                print(f'\tRunning SM model: {full_formula} with {self.levels_list[0]}')
                if self.df_bisect_col is not None:
                    print(f'\tBisecting df on column: {self.df_bisect_col} to keep {self.df_bisect_cat}')
                print('='*20)
                print('\n')
                print(f'Reference categories dict for {self.hypothesis_num.capitalize()} {self.dv}:')
                print('='*20)
                pprint.pprint(self.reference_cat_dict)
                print('\n')
                print('='*20)
                pprint.pprint(sm_model.summary())
                if self.ols_only:
                    print('\n')
                    print('='*20)
                    print(f'ANOVAs model {i} for {self.hypothesis_num.capitalize()}:')
                    print('='*20)
                    pprint.pprint(sm.stats.anova_lm(sm_model, typ=2))
                print('\n\n')
                if self.plot_enabled and i == len(self.sm_formula_list) - 1:
                    self.plot_sm_model(sm_model)
                    print('\n\n')

        return self.sm_model_results_list

    def make_results_header_str(self, i):
        if self.df_bisect_col is not None:
            return f'B (Model {i} {self.df_bisect_col} = {self.df_bisect_cat})'
        else:
            return f'B (Model {i})'

    def get_model_results(self):
        if not self.ols_only:
            self.model_results = self.fitted_models[self.dv][self.hypothesis_num]['pymer'] = self.fit_stepwise_models()
            if self.sm_check:
                try:
                    self.fitted_models[self.dv][self.hypothesis_num]['sm'] = self.fit_stepwise_sm_models()
                except Exception as e:
                    print(f'Error fitting OLS models: {e}')
        if self.ols_only:
            self.model_results = self.fitted_models[self.dv][self.hypothesis_num]['sm'] = self.fit_stepwise_sm_models()

        return self.model_results

    def get_result_df(self):
        self.result_df = pd.DataFrame(
            columns=['Predictor'] + [
                f'B (Model {i} - {self.df_bisect_col} = {self.df_bisect_cat})'
                if self.df_bisect_col is not None
                else f'B (Model {i})'
                for i in range(len(self.model_results))
            ],
        )
        return self.result_df

    def calculate_sm_lr_stat(self, model1, model2):
        lr_stat, df_diff, p_value = model1.compare_lr_test(model2)
        df_lr_stat = pd.DataFrame(
            {
                'lr_stat': lr_stat,
                'df_diff': df_diff,
                'p_value': p_value,
            }, index=[0],
        )
        return df_lr_stat

    def save_comparison_df(self, df_model_comparison):
        df_model_comparison_save_path = shorten_file_path(
            f'{hypotheses_model_save_path}{self.hypothesis_num}_{self.dv}_df_model_comparison',
        )
        df_model_comparison.to_pickle(f'{df_model_comparison_save_path}.pkl')
        df_model_comparison.to_csv(f'{df_model_comparison_save_path}.csv', index=False)
        df_model_comparison.to_excel(f'{df_model_comparison_save_path}.xlsx', index=False)

    def compare_models(self):
        if len(self.model_results) > 1:
            df_model_comparison = pd.DataFrame()
            for model1, model2 in tqdm.tqdm(itertools.combinations(self.model_results, 2)):
                if not self.ols_only:
                    model1_formula = model1.formula
                    model2_formula = model2.formula
                    model_comparison = pymer4.stats.lrt([model1, model2], refit=True)
                elif self.ols_only:
                    model1_formula = model1.model.formula
                    model2_formula = model2.model.formula
                    model_comparison = self.calculate_sm_lr_stat(model1, model2)
                    model_comparison['model'] = f'{model1_formula}\n{model2_formula}'
                    model_comparison = model_comparison.set_index('model')
                df_model_comparison = pd.concat([df_model_comparison, model_comparison], ignore_index=True)
                print('='*20)
                print(f'Comparing models:\n\t{model1_formula}\n\t\tvs.\n\t{model2_formula}')
                print('='*20)
                pprint.pprint(model_comparison)
                print('\n\n')
            if self.model_save_enabled:
                self.save_comparison_df(df_model_comparison)

    def get_predictors_list(self):
        for i, model in enumerate(self.model_results):
            if not self.ols_only:
                self.coefs = model.coefs.index.tolist()
            elif self.ols_only:
                self.coefs = model.params.index.tolist()

            self.intercept = []
            self.main_predictors = []
            self.covariates = []
            self.moderators = []
            self.iv_interaction_predictors = []
            self.mod_interaction_predictors = []

            for p in self.coefs:
                if 'Intercept' in p:
                    self.intercept.append(p)
                if not (
                    ':' in p
                    or ' x ' in p
                ):
                    if any(iv in p for iv in self.ivs_list):
                        self.main_predictors.append(p)
                    if any(cov in p for cov in self.cols_cov_list):
                        self.covariates.append(p)
                    if any(mod in p for mod in self.moderators_list):
                        self.moderators.append(p)
                elif (
                    ':' in p
                    or ' x ' in p
                ):
                    # Determine how many IVs appear in the term
                    iv_count = sum(p.count(iv) for iv in self.ivs_list)
                    has_mod = any(mod in p for mod in self.moderators_list)
                    # Case 1: Pure IV×IV interaction (no moderator)
                    if iv_count > 1 and not has_mod:
                        if self.iv_interactions:
                            if p not in self.iv_interaction_predictors:
                                self.iv_interaction_predictors.append(p)
                    # Case 2: IV × Moderator (or multiple moderators) but only one IV
                    elif has_mod and iv_count == 1:
                        if p not in self.mod_interaction_predictors:
                            self.mod_interaction_predictors.append(p)
                    # All other cases (including iv_count >1 with moderators) are skipped

        self.predictors = self.intercept + self.main_predictors

        if len(self.moderators_list) > 0:
            self.predictors += self.moderators
        if len(self.ivs_list) > 1:
            self.predictors += self.iv_interaction_predictors
        if len(self.moderators_list) > 0:
            self.predictors += self.mod_interaction_predictors

        return self.predictors

    def assign_significance_star(self, p_value):
        if p_value < 0.001:
            significance_star = '***'
        elif p_value < 0.01:
            significance_star = '**'
        elif p_value < 0.05:
            significance_star = '*'
        else:
            significance_star = ''
        return significance_star

    def multilevel_estimates(self, predictor, i, model, row):
        if predictor in model.coefs.index:
            b_value = model.coefs.loc[predictor, 'Estimate']
            se_value = model.coefs.loc[predictor, 'SE']
            p_value = model.coefs.loc[predictor, 'P-val']
            significance_star = self.assign_significance_star(p_value)
            row[self.make_results_header_str(i)] = f'{b_value:.{self.decimals}f}{significance_star} ({se_value:.{self.decimals}f})'
        else:
            row[self.make_results_header_str(i)] = ''
        return row

    def sm_ols_estimates(self, predictor, i, model, row):
        if predictor in model.params.index:
            b_value = model.params.loc[predictor]
            se_value = model.bse.loc[predictor]
            p_value = model.pvalues.loc[predictor]
            significance_star = self.assign_significance_star(p_value)
            row[self.make_results_header_str(i)] = f'{b_value:.{self.decimals}f}{significance_star} ({se_value:.{self.decimals}f})'
        else:
            row[self.make_results_header_str(i)] = ''
        return row

    def get_estimates(self):
        # Estimates, standard errors, and significance stars
        for predictor in self.predictors:
            row = {'Predictor': predictor}
            for i, model in enumerate(self.model_results):
                if not self.ols_only:
                    row = self.multilevel_estimates(predictor, i, model, row)
                elif self.ols_only:
                    row = self.sm_ols_estimates(predictor, i, model, row)
            self.result_df = pd.concat([self.result_df, pd.DataFrame([row])], ignore_index=True).reset_index(drop=True)
        return self.result_df

    def get_sm_f_stats(self):
        f_stat_row = {'Predictor': 'F-statistic'}
        for i, model in enumerate(self.model_results):
            f_stat = float(model.fvalue)
            significance_star = self.assign_significance_star(model.f_pvalue)
            f_stat_row[self.make_results_header_str(i)] = f'{f_stat:.{self.decimals}f}{significance_star}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([f_stat_row])], ignore_index=True)
        return self.result_df

    def get_model_df(self):
        # Get degrees of freedom
        df1_row = {'Predictor': 'DF Model'}
        df2_row = {'Predictor': 'DF Residual'}
        for i, model in enumerate(self.model_results):
            df1_row[self.make_results_header_str(i)] = f'{model.df_model:.0f}'
            df2_row[self.make_results_header_str(i)] = f'{model.df_resid:.0f}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([df1_row, df2_row])], ignore_index=True)
        return self.result_df

    def get_group_variance(self):
        group_var_row = {'Predictor': 'Group Variance'}
        for i, model in enumerate(self.model_results):
            group_var = float(model.ranef_var['Var'][0]) if not model.ranef_var['Var'].empty else np.nan
            group_var_row[self.make_results_header_str(i)] = f'{group_var:.{self.decimals}f}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([group_var_row])], ignore_index=True)
        return self.result_df

    def get_icc(self):
        # ICC
        icc_row = {'Predictor': 'ICC'}
        for i, model in enumerate(self.model_results):
            group_var = float(model.ranef_var['Var'][0]) if not model.ranef_var['Var'].empty else np.nan
            residual_var = float(model.ranef_var.loc['Residual', 'Var'])
            total_var = float(group_var + residual_var) if group_var + residual_var > 0 else np.nan
            icc = float(group_var / total_var)
            icc_row[self.make_results_header_str(i)] = f'{icc:.{self.decimals}f}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([icc_row])], ignore_index=True)
        return self.result_df

    def get_aic(self):
        # AIC
        aic_row = {'Predictor': 'AIC'}
        for i, model in enumerate(self.model_results):
            if not self.ols_only:
                aic_row[self.make_results_header_str(i)] = f'{model.AIC:.{self.decimals}f}'
            elif self.ols_only:
                aic_row[self.make_results_header_str(i)] = f'{model.aic:.{self.decimals}f}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([aic_row])], ignore_index=True)
        return self.result_df

    def get_bic(self):
        # BIC
        bic_row = {'Predictor': 'BIC'}
        for i, model in enumerate(self.model_results):
            if not self.ols_only:
                bic_row[self.make_results_header_str(i)] = f'{model.BIC:.{self.decimals}f}'
            elif self.ols_only:
                bic_row[self.make_results_header_str(i)] = f'{model.bic:.{self.decimals}f}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([bic_row])], ignore_index=True)
        return self.result_df

    def get_log_likelihood(self):
        # Log-likelihood
        log_likelihood_row = {'Predictor': 'Log-Likelihood'}
        for i, model in enumerate(self.model_results):
            if not self.ols_only:
                log_likelihood_row[self.make_results_header_str(i)] = f'{model.logLike:.{self.decimals}f}'
            elif self.ols_only:
                log_likelihood_row[self.make_results_header_str(i)] = f'{model.llf:.{self.decimals}f}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([log_likelihood_row])], ignore_index=True)
        return self.result_df

    def get_pymer_pseudo_r2(self, model):
        fixed_var = np.var(model.fits)
        group_var = 0.0
        for idx in model.ranef_var.index:
            if idx != 'Residual':
                group_var += float(model.ranef_var.loc[idx, 'Var'])
        residual_var = float(model.ranef_var.loc['Residual', 'Var'])

        total_var = float(fixed_var + group_var + residual_var) if fixed_var + group_var + residual_var > 0 else np.nan
        r2_m = float(fixed_var / total_var)
        r2_c = float((fixed_var + group_var) / total_var)
        return r2_m, r2_c

    def get_pseudo_r2(self):
        r2_row = {'Predictor': 'R²'}
        marginal_pseudo_r2_row = {'Predictor': 'Pseudo-R² (Marginal)'}
        conditional_pseudo_r2_row = {'Predictor': 'Pseudo-R² (Conditional)'}
        for i, model in enumerate(self.model_results):
            if not self.ols_only:
                r2_m, r2_c = self.get_pymer_pseudo_r2(model)
                marginal_pseudo_r2_row[self.make_results_header_str(i)] = f'{r2_m:.{self.decimals}f}'
                conditional_pseudo_r2_row[self.make_results_header_str(i)] = f'{r2_c:.{self.decimals}f}'
            elif self.ols_only:
                r2_row[self.make_results_header_str(i)] = f'{model.rsquared:.{self.decimals}f}'
        if not self.ols_only:
            self.result_df = pd.concat(
                [self.result_df, pd.DataFrame([marginal_pseudo_r2_row, conditional_pseudo_r2_row])],
                ignore_index=True,
            )
        if self.ols_only:
            self.result_df = pd.concat([self.result_df, pd.DataFrame([r2_row])], ignore_index=True)
        return self.result_df

    def get_n_obs(self):
        # Number of observations and groups
        obs_rows = {'Predictor': 'N (Obs)'}
        for i, model in enumerate(self.model_results):
            if not self.ols_only:
                obs_rows[self.make_results_header_str(i)] = f'{int(len(model.data))}'
            elif self.ols_only:
                obs_rows[self.make_results_header_str(i)] = f'{int(model.nobs)}'
        self.result_df = pd.concat([self.result_df, pd.DataFrame([obs_rows])], ignore_index=True)
        return self.result_df

    def get_n_groups(self):
        # Number of observations and groups
        groups_rows = {'Predictor': 'N (Groups)'}
        for i, model in enumerate(self.model_results):
            n_groups = int(next(iter(model.grps.values())))
            groups_rows[self.make_results_header_str(i)] = n_groups
        self.result_df = pd.concat([self.result_df, pd.DataFrame([groups_rows])], ignore_index=True)
        return self.result_df

    def get_n_levels(self):
        # Number of levels
        levels_rows = {'Predictor': 'N (Levels)'}
        for i, model in enumerate(self.model_results):
            levels_rows[self.make_results_header_str(i)] = len(self.levels_list)
        self.result_df = pd.concat([self.result_df, pd.DataFrame([levels_rows])], ignore_index=True)
        return self.result_df

    def add_p_value_note(self):
        p_value_note_row = {'Predictor': f'* p < 0.05, ** p < 0.01, *** p < 0.001'}
        self.result_df = pd.concat([self.result_df, pd.DataFrame([p_value_note_row])], ignore_index=True)
        return self.result_df

    def add_controls_note(self):
        control_variable_str = []
        for i, col in enumerate(self.cols_cov_list):
            if self.cond(col):
                cats = list(self.df[col].cat.categories)
                measurement_level = 'Binary' if len(cats) == 2 else 'Categorical'
                cats_joined = " vs. ".join(rename_predictor_name(cat) for cat in cats)
                new_str = f'({measurement_level}; {cats_joined})'
                control_variable_str.append(f'({i + 1}) {col} {new_str}')
        if any(not self.cond(col) for col in self.cols_cov_list):
            control_variable_str.append('the abovementioned covariates')

        if len(control_variable_str) == 2:
            formatted_controls = f'{control_variable_str[0]} and {control_variable_str[1]}'
        elif len(control_variable_str) > 2:
            formatted_controls = ', '.join(control_variable_str[:-1]) + f', and {control_variable_str[-1]}'
        else:
            formatted_controls = control_variable_str[0] if control_variable_str else ''

        if control_variable_str:
            controls_note_row = {'Predictor': f'Note. Control variable(s) = {formatted_controls}'}

            self.result_df = pd.concat([self.result_df, pd.DataFrame(data=[controls_note_row])], ignore_index=True)
            return self.result_df
        else:
            return self.result_df

    def add_hypothesis_num_dv(self):
        hypothesis_num_dv_row = [
            f'Dependent variable: {rename_predictor_name(self.dv)}',
        ]
        self.result_df.columns = pd.MultiIndex.from_product(
            [hypothesis_num_dv_row, self.result_df.columns],
        )
        return self.result_df

    def save_original_model_summary(self):
        for i, model in enumerate(self.model_results):
            model_to_save = model
            if not self.ols_only:
                save_path = f'{hypotheses_model_save_path}{self.hypothesis_num}_{self.dv}_model{i}_summary'
                if self.df_bisect_col is not None:
                    save_path += f'_{self.df_bisect_cat}'
                model_to_save.coefs = model_to_save.coefs.rename(index=rename_predictor_name)
                model_to_save.coefs.to_csv(f'{save_path}.csv')
                model_to_save.coefs.to_excel(f'{save_path}.xlsx')
            elif self.ols_only:
                save_path = f'{hypotheses_model_save_path}{self.hypothesis_num}_{self.dv}_sm_model{i}_summary'
                if self.df_bisect_col is not None:
                    save_path += f'_{self.df_bisect_cat}'
                model_to_save.params = model_to_save.params.rename(index=rename_predictor_name)
                df_model = pd.read_csv(StringIO('\n'.join(re.sub(r'\((.*?)\)', lambda m: '(' + m.group(1).replace(',', ';') + ')', line) for line in model_to_save.summary().tables[1].as_csv().splitlines())), skipinitialspace=True, engine='python', index_col=0)
                df_model.to_csv(f'{save_path}.csv')
                df_model.to_excel(f'{save_path}.xlsx')

    def get_markdown_table(self):
        markdown_header_str = f'# Stepwise Mixed-Effects Model Results for {rename_predictor_name(self.dv)}\n\n'
        if self.df_bisect_col is not None:
            markdown_header_str += f'Only for {rename_predictor_name(self.df_bisect_col)} = {rename_predictor_name(self.df_bisect_cat)}\n\n'
        self.markdown_table = (
            markdown_header_str +
            self.result_df.to_markdown(index=False, tablefmt='github', floatfmt=f'.{self.decimals}f')
        )
        return self.markdown_table

    def get_latex_table(self):
        latex_caption_str = f'Stepwise Mixed-Effects Model Results for {rename_predictor_name(self.dv)}'
        if self.df_bisect_col is not None:
            latex_caption_str += f' - Only for {rename_predictor_name(self.df_bisect_col)} = {rename_predictor_name(self.df_bisect_cat)}'
        self.latex_table = (
            '\\begin{table}[ht]\n'
            '\\centering\n'
            f'\\caption{{latex_caption_str}}\n'
            f'\\label{{tab:{rename_predictor_name(self.dv)}}}\n'
            '\\resizebox{\\textwidth}{!}{%\n' +
            self.result_df.to_latex(index=False, escape=True, na_rep='') +
            '}\n\\end{table}'
        )
        return self.latex_table

    def save_tables(self):
        save_path = f'{hypotheses_table_save_path}{self.hypothesis_num}_{self.dv}_results_table'
        if self.df_bisect_col is not None:
            save_path += f'_{self.df_bisect_cat}'
        print('='*20)
        print(f'Saving tables for {rename_predictor_name(self.dv)} {self.hypothesis_num}')
        print('='*20)
        print('Saving Markdown table...')
        with open(f'{save_path}.md', 'w') as f:
            f.write(self.markdown_table)
        print('Markdown table saved successfully!')
        print('\n')
        print('Saving LaTeX table...')
        with open(f'{save_path}.tex', 'w') as f:
            f.write(self.latex_table)
        print('LaTeX table saved successfully!')
        print('\n')
        print('Saving pickled table...')
        self.result_df.to_pickle(f'{save_path}.pkl')
        print('Pickled table saved successfully!')
        print('\n')
        print('Saving CSV table...')
        self.result_df.to_csv(f'{save_path}.csv', index=False)
        print('CSV table saved successfully!')
        print('\n')
        print('Saving Excel table...')
        self.result_df.to_excel(f'{save_path}.xlsx')
        print('Excel table saved successfully!')
        print('\n')

    def print_results_table(self):
        print('='*20)
        print(f'\tMultilevel model results for {rename_predictor_name(self.dv)} {self.hypothesis_num}:')
        if self.df_bisect_col is not None:
            print(f'\tOnly for {rename_predictor_name(self.df_bisect_col)} = {rename_predictor_name(self.df_bisect_cat)}')
        print('='*20)
        pprint.pprint(self.result_df)
        print('\n')

    def fix_result_df(self):
        self.result_df = self.get_estimates()
        self.result_df = self.get_n_obs()
        if not self.ols_only:
            self.result_df = self.get_n_groups()
            self.result_df = self.get_n_levels()
        if not self.ols_only:
            self.results_df = self.get_group_variance()
        elif self.ols_only:
            self.result_df = self.get_sm_f_stats()
            self.result_df = self.get_model_df()
        self.result_df = self.get_pseudo_r2()
        if not self.ols_only:
            self.result_df = self.get_icc()
        self.result_df = self.get_aic()
        self.result_df = self.get_bic()
        self.result_df = self.get_log_likelihood()
        self.result_df = self.add_controls_note()
        self.result_df = self.add_p_value_note()
        self.result_df['Predictor'] = self.result_df['Predictor'].apply(rename_predictor_name).replace({'Intercept': 'Intercept (Group)'})
        self.result_df = self.add_hypothesis_num_dv()
        return self.result_df

    def make_results_table(self):
        self.order_cat_cols()
        self.descriptives()
        self.reference_cat_dict = self.make_reference_cat_dict()
        self.model_results = self.get_model_results()
        self.result_df = self.get_result_df()
        if self.model_compare_enabled:
            self.compare_models()
        self.predictors = self.get_predictors_list()
        self.result_df = self.fix_result_df()
        self.markdown_table = self.get_markdown_table()
        self.latex_table = self.get_latex_table()
        if self.save_enabled:
            self.save_tables()
        if self.model_save_enabled:
            self.save_original_model_summary()
        if self.print_summary:
            self.print_results_table()

        return self.result_df

# %%
