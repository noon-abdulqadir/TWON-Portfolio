import os # type:ignore # isort:skip # fmt:skip # noqa # nopep8
import sys # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path # type:ignore # isort:skip # fmt:skip # noqa # nopep8

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
from setup_module.imports import * # type:ignore # isort:skip # fmt:skip # noqa # nopep8

# %%
# Get ISO sectors
@dataclass(repr=True)
class SectorsData:
    '''
    This class is used to get the sectors data from the 1. CBS notebook.
    '''

    def get_df_sectors_all(self) -> pd.DataFrame:
        self.df_sectors_all_path = f'{table_save_path}Sectors Output from script.pkl'
        try:
            self.df_sectors_all = pd.read_pickle(self.df_sectors_all_path)
        except FileNotFoundError:
            # sectors_notebook = '\\'.join(f'{scraped_data}1. Sector_Data/1. CBS.ipynb')
            # %run $sectors_notebook import df_sectors_all
            # self.df_sectors_all = df_sectors_all
            exec(open(f'{scraped_data}1. Sector_Data/cbs.py').read())
            self.df_sectors_all = df_sectors_all
        return self.df_sectors_all

    def clean_df_sectors_all(self) -> pd.DataFrame:
        self.df_sectors_all = self.get_df_sectors_all()
        self.df_sectors_all.columns = [
            '_'.join(col).strip()
            if 'Sector Titles' not in col
            and 'Total Workforce' not in col
            else col[-1]
            for col in self.df_sectors_all.columns
        ]
        self.df_sectors_all = self.df_sectors_all.rename(
            columns={
                'Keywords': 'Search Keyword',
                'Code': 'Sector Code',
                'Gender_Sectoral Gender Segregation_Dominant Category': 'Gender',
                'Age_Sectoral Age Segregation_Dominant Category': 'Age',
                'n': 'Sector_n',
            },
        )
        self.df_sectors_all = self.df_sectors_all.rename(
            columns={
                element.strip(): re.sub(r' \(\W*45 years\)', '', element).strip()
                for element in self.df_sectors_all.columns.tolist()
            },
        )

        return self.df_sectors_all

    def get_sectors_names_list(self) -> list:
        self.sectors_names_list = self.df_sectors_all['ISIC 4 Sector Name'].values.tolist()
        self.sectors_names_list = [
            element.strip()
            for element in self.sectors_names_list
            if str(element) not in ['nan', 'Total (excluding A-U)', 'All economic activities']
        ]
        with open(f'{data_dir}sectors_names_list.txt', 'w') as f:
            f.write('\n'.join(self.sectors_names_list))
        return self.sectors_names_list

    def get_sectors_codes_list(self) -> list:
        self.sectors_codes_list = self.df_sectors_all['Sector Code'].values.tolist()
        self.sectors_codes_list = [
            code.strip()
            for code in self.sectors_codes_list
            if str(code) not in ['nan', 'A-U']
        ]
        with open(f'{data_dir}sectors_codes_list.txt', 'w') as f:
            f.write('\n'.join(self.sectors_codes_list))
        return self.sectors_codes_list

    def get_sectors_codes_dict(self) -> dict:
        self.sectors_codes_dict = self.df_sectors_all.set_index('Sector Code')['ISIC 4 Sector Name'].to_dict()
        self.sectors_codes_dict = {
            code.strip(): sector.strip()
            for code, sector in self.sectors_codes_dict.items()
            if str(code) not in ['nan', 'A-U']
        }
        with open(f'{data_dir}sectors_codes_dict.json', 'w') as f:
            json.dump(self.sectors_codes_dict, f)
        return self.sectors_codes_dict

    def get_search_keywords_list(self) -> list:
        self.search_keywords_list = list(
            {
                search_keyword.strip().lower()
                for search_keyword in list(set(self.df_sectors_all.explode('Search Keyword', ignore_index=True)['Search Keyword'].values.tolist()))
                if isinstance(search_keyword, str)
                and search_keyword not in nan_list[:] + non_whitespace_nan_list[:]
                and len(search_keyword) > 0
            },
        )
        with open(f'{data_dir}search_keywords_list.txt', 'w') as f:
            f.write('\n'.join(self.search_keywords_list))
        return self.search_keywords_list

    def print_info(func: typing.Callable) -> typing.Callable:
        def wrapper(self, *args, **kwargs) -> typing.Callable:
            (
                self.df_sectors_all,
                self.sectors_names_list,
                self.sectors_codes_list,
                self.sectors_codes_dict,
                self.search_keywords_list,
            ) = func(self, *args, **kwargs)
            print('='*20)
            print(f'Number of Sectors: {len(self.sectors_names_list)}')
            print(f'Sector Names:\n{self.sectors_names_list}', end='\n')
            print(f'Sector Codes:\n{self.sectors_codes_list}', end='\n')
            print(f'Sector Codes Dict:\n{self.sectors_codes_dict}', end='\n')
            print('-'*20)
            print(f'Number of Search Keywords: {len(self.search_keywords_list)}')
            print(f'Search Keywords:\n{self.search_keywords_list}', end='\n')
            print('='*20)
            return (
                self.df_sectors_all,
                self.sectors_names_list,
                self.sectors_codes_list,
                self.sectors_codes_dict,
                self.search_keywords_list,
            )
        return wrapper

    @print_info
    def get_sector_data(self) -> tuple[pd.DataFrame, list, list, dict, list]:
        self.clean_df_sectors_all()
        self.get_sectors_names_list()
        self.get_sectors_codes_list()
        self.get_sectors_codes_dict()
        self.get_search_keywords_list()

        return self.df_sectors_all, self.sectors_names_list, self.sectors_codes_list, self.sectors_codes_dict, self.search_keywords_list

# %%
# Get ISO language codes
class LanguageCodes:

    def __init__(
        self,
    ) -> pd.DataFrame:
        pass

    def get_temp1(self, language_codes_path: str = None, not_nan_col: str = None) -> pd.DataFrame:
        if language_codes_path is None:
            language_codes_path = f'{fb_found_data_path}iso_639-1_datahub.csv'
        if not_nan_col is None:
            not_nan_col = 'iso-639-2/B'
        try:
            temp1 = pd.read_csv(language_codes_path)
        except FileNotFoundError:
            temp1 = pd.read_csv('https://datahub.io/core/language-codes/r/language-codes-full.csv')
            temp1.to_csv(language_codes_path, index=False)

        temp1 = temp1.rename(
            columns={
                'English': 'language_name', 'French': 'language_french_name', 'alpha2': 'iso-639-1', 'alpha3-b': 'iso-639-2/B', 'alpha3-t': 'iso-639-2/T',
            },
        ).dropna(subset=[not_nan_col]).reset_index(drop=True)
        temp1 = temp1.drop(temp1.loc[temp1[not_nan_col].isin(['cnr', 'qaa-qtz'])].index, axis='index').reset_index(drop=True)
        for idx, row in temp1.iterrows():
            alpha3_b = str(row[not_nan_col]).lower().strip()
            with contextlib.suppress(KeyError):
                iso_obj = iso639.languages.get(part2b=alpha3_b)
                temp1.loc[idx, 'language_name'] = iso_obj.name.lower().strip()
                temp1.loc[idx, 'iso-639-1'] = iso_obj.part1.lower().strip()
                temp1.loc[idx, 'iso-639-2'] = iso_obj.part3.lower().strip()
                temp1.loc[idx, 'iso-639-2/T'] = iso_obj.part2t.lower().strip()
                temp1.loc[idx, 'iso-639-5'] = iso_obj.part5.lower().strip()
                temp1.loc[idx, 'other_language_names'] = str(list(iso_obj.names))
        return temp1[['language_name', 'language_french_name', 'iso-639-1', 'iso-639-2', 'iso-639-2/B', 'iso-639-2/T', 'iso-639-5', 'other_language_names']].sort_values(not_nan_col).reset_index(drop=True)

    def get_temp2(self, language_codes_path: str = None, not_nan_col: str = None) -> pd.DataFrame:
        if language_codes_path is None:
            language_codes_path = f'{fb_found_data_path}iso_639-1_github.csv'
        if not_nan_col is None:
            not_nan_col = 'iso-639-2'
        try:
            temp2 = pd.read_csv(language_codes_path)
        except FileNotFoundError:
            temp2 = pd.read_csv('https://raw.githubusercontent.com/haliaeetus/iso-639/master/data/iso_639-1.csv')
            temp2.to_csv(language_codes_path, index=False)

        temp2 = temp2.rename(
            columns={
                'name': 'language_name_capitalized', 'nativeName': 'language_native_name',
            } | {col: f'iso-{col.strip()}' for col in temp2.columns if '639' in str(col)},
        ).dropna(subset=[not_nan_col]).reset_index(drop=True)
        for idx, row in temp2.iterrows():
            alpha2 = str(row[not_nan_col]).lower().strip()
            with contextlib.suppress(KeyError):
                iso_obj = iso639.languages.get(part3=alpha2)
                temp2.loc[idx, 'iso-639-1'] = iso_obj.part1.lower().strip()
                temp2.loc[idx, 'iso-639-2/B'] = iso_obj.part2b.lower().strip()
                temp2.loc[idx, 'iso-639-2/T'] = iso_obj.part2t.lower().strip()
                temp2.loc[idx, 'iso-639-5'] = iso_obj.part5.lower().strip()
                temp2.loc[idx, 'other_language_names'] = str(list(iso_obj.names))
        return temp2[['language_name_capitalized', 'language_native_name', 'family', 'iso-639-1', 'iso-639-2', 'iso-639-2/B', 'iso-639-2/T', 'iso-639-5', 'other_language_names']].sort_values(not_nan_col).reset_index(drop=True)

    def get_df_language_codes(self) -> pd.DataFrame:
        self.df_language_codes = self.get_temp1().merge(self.get_temp2(), on='iso-639-2', how='outer', suffixes=('_temp1', '_temp2'))
        self.df_language_codes['iso-639-1'] = self.df_language_codes['iso-639-1_temp1'].fillna(self.df_language_codes['iso-639-1_temp2'])
        self.df_language_codes['iso-639-2/B'] = self.df_language_codes['iso-639-2/B_temp1'].fillna(self.df_language_codes['iso-639-2/B_temp2'])
        self.df_language_codes['iso-639-2/T'] = self.df_language_codes['iso-639-2/T_temp1'].fillna(self.df_language_codes['iso-639-2/T_temp2'])
        self.df_language_codes['iso-639-5'] = self.df_language_codes['iso-639-5_temp1'].fillna(self.df_language_codes['iso-639-5_temp2'])
        self.df_language_codes['other_language_names'] = self.df_language_codes['other_language_names_temp1'].fillna(self.df_language_codes['other_language_names_temp2'])
        self.df_language_codes = self.df_language_codes.drop(columns=['iso-639-1_temp1', 'iso-639-1_temp2', 'iso-639-2/B_temp1', 'iso-639-2/B_temp2', 'iso-639-2/T_temp1', 'iso-639-2/T_temp2', 'iso-639-5_temp1', 'iso-639-5_temp2', 'other_language_names_temp1', 'other_language_names_temp2'])
        self.df_language_codes['iso-639-2'] = self.df_language_codes['iso-639-2'].fillna(self.df_language_codes['iso-639-2/B']).fillna(self.df_language_codes['iso-639-2/T'])
        self.df_language_codes['language_name'] = self.df_language_codes['language_name'].fillna(self.df_language_codes['language_name_capitalized'].str.lower())
        with contextlib.suppress(KeyError):
            self.df_language_codes['language_name'] = self.df_language_codes['language_name'].fillna(
                self.df_language_codes['iso-639-2'].apply(
                    lambda x: iso639.languages.get(part3=x).name.lower().strip()
                    if isinstance(x, str) and len(x) != 0
                    else np.nan,
                ),
            )
        self.df_language_codes['language_code'] = self.df_language_codes['iso-639-1']
        with contextlib.suppress(KeyError):
            self.df_language_codes['language_code'] = self.df_language_codes['language_code'].fillna(
                self.df_language_codes['iso-639-2'].apply(
                    lambda x: iso639.languages.get(part3=x).part1.lower().strip()
                    if isinstance(x, str) and len(x) != 0
                    else np.nan,
                ),
            )
        self.df_language_codes['language_name_capitalized'] = self.df_language_codes.apply(lambda x: x['language_name_capitalized'] if isinstance(x['language_name_capitalized'], str) and len(x['language_name_capitalized']) != 0 else x['language_name'].strip().title(), axis='columns')
        self.df_language_codes = self.df_language_codes[
            ['language_name', 'language_code', 'language_name_capitalized', 'language_french_name', 'language_native_name', 'other_language_names', 'iso-639-1', 'iso-639-2', 'iso-639-2/B', 'iso-639-2/T', 'iso-639-5']
        ].sort_values('iso-639-2').reset_index(drop=True)
        self.df_language_codes.to_csv(f'{fb_found_data_path}ISO 639 Language Codes.csv', index=False)
        self.df_language_codes.to_pickle(f'{fb_found_data_path}ISO 639 Language Codes.pkl')
        return self.df_language_codes

# %%
# Get ISO All codes Excel
class AllCodesExcel:

    def __init__(
        self,
        file_save_path: str = None,
        caption: str = None,
    ) -> None:
        self.file_save_path = file_save_path
        self.caption = caption
        if self.file_save_path is None:
            self.file_save_path = f'{data_dir}All Codes'
        if self.caption is None:
            self.caption = 'Countries and respective ISO codes'

    def save_codes_table_all(self):
        # Save other formats
        self.df_codes_table.to_csv(f'{self.file_save_path}.csv', index=False)
        self.df_codes_table.to_pickle(f'{self.file_save_path}.pkl')
        self.df_codes_table.to_markdown(f'{self.file_save_path}.md')
        with pd.option_context('max_colwidth', 10000000000):
            self.df_codes_table.style.to_latex(
                f'{self.file_save_path}.tex',
                convert_css=True,
                environment='longtable',
                hrules=True,
                # escape=True,
                # multicolumn=True,
                multicol_align='c',
                position='H',
                caption=self.caption,
            )

    def add_df_codes_table_header(
        self,
        col_width_dict: dict = None,
        startrow: int = None,
        startcol: int = None,
    ) -> None:
        if col_width_dict is None:
            col_width_dict = {
                'Count ID': 1.5*5,
                'Country': 3.5*5,
                '2-letter Code: ISO3166-1-Alpha-2': 3.5*5,
                '3-letter Code: ISO3166-1-Alpha-3': 3.5*5,
                'Numeric Code: ISO3166-1-numeric': 3.5*5,
            }
        header_formats = {'bold': False, 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'center', 'top': True, 'bottom': True, 'left': False, 'right': False, 'text_wrap': True}

        for col_name, col_width in col_width_dict.items():
            row_to_write = startrow
            col_to_write = startcol + self.df_codes_table.columns.get_loc(col_name)
            self.worksheet.set_column(col_to_write, col_to_write, col_width)
            two_line_col_name = f'{col_name.split(":")[0]}:\n{col_name.split(":")[1].strip()}' if ':' in col_name else col_name
            self.worksheet.write(row_to_write, col_to_write, two_line_col_name, self.workbook.add_format(header_formats))

    def add_df_codes_table_body(
        self,
        startrow: int = None,
        startcol: int = None,
    ) -> None:
        num = [col_num for col_num, value in enumerate(self.df_codes_table.columns.values) if 'Numeric Code' in value]
        letter = [col_num for col_num, value in enumerate(self.df_codes_table.columns.values) if 'letter Code' in value]

        row_idx, col_idx = self.df_codes_table.shape
        for c, r in tqdm_product(range(col_idx), range(row_idx)):
            row_to_write = startrow + 1 + r # 1 is for the hidden empty column under the header
            col_to_write = startcol + c # 1 is for index
            body_formats = {'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'left': False, 'right': False, 'text_wrap': True}

            if r == row_idx-1:
                body_formats |= {'bottom': True}

            if c in num:
                body_formats |= {'num_format': '0', 'align': 'center'}

            if c in letter:
                body_formats |= {'align': 'center'}

            self.worksheet.write(row_to_write, col_to_write, self.df_codes_table.iloc[r, c], self.workbook.add_format(body_formats))

    def save_df_codes_table(
        self,
        sheet_name: str = None,
        caption: str = None,
        startrow: int = None,
        startcol: int = None,
    ):
        if sheet_name is None:
            sheet_name = 'All'
        if caption is None:
            caption = 'Countries and respective ISO codes'
        if startrow is None:
            startrow = 2
        if startcol is None:
            startcol = 1

        self.df_codes_table = self.make_df_codes_table()
        self.save_codes_table_all()

        # Define last rows and cols locs
        endrow = startrow + self.df_codes_table.shape[0]
        endcol = startcol + self.df_codes_table.shape[1] - 1

        # Save excel
        writer = pd.ExcelWriter(f'{self.file_save_path}.xlsx')
        self.df_codes_table.to_excel(writer, sheet_name=sheet_name, merge_cells=True, index=False, startrow=startrow, startcol=startcol)
        self.workbook = writer.book
        self.worksheet = writer.sheets[sheet_name]

        # Create title row
        self.worksheet.merge_range(1, startcol, 1, endcol, self.caption, self.workbook.add_format({'bold': False, 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'left', 'top': True, 'bottom': True, 'left': False, 'right': False}))

        # Format headers
        self.add_df_codes_table_header(startrow=startrow, startcol=startcol)

        # Format body
        self.add_df_codes_table_body(startrow=startrow, startcol=startcol)

        # Add Note
        note_format = {'italic': True, 'font_name': 'Times New Roman', 'font_size': 10, 'font_color': 'black', 'align': 'left', 'left': False, 'right': False}
        self.worksheet.merge_range(endrow + 1, startcol, endrow + 1, endcol, 'Source: Datahub', self.workbook.add_format(note_format))
        writer.close()

# %%
# Get ISO All codes


class AllCodes(LanguageCodes, AllCodesExcel):

    def __init__(
        self,
        country_names_list: list = None,
        eu_country_names_list: list = None,
        use_saved_country_codes: bool = None,
        use_eu_country_codes: bool = None,
        use_og_country_codes: bool = None,
        country_names_fix_dict: dict = None,
        currency_dict: dict = None,
    ) -> None:

        LanguageCodes.__init__(self)
        AllCodesExcel.__init__(self)

        self.country_names_list: list = country_names_list
        self.eu_country_names_list: list = eu_country_names_list
        self.use_saved_country_codes: bool = use_saved_country_codes
        self.use_eu_country_codes: bool = use_eu_country_codes
        self.use_og_country_codes: bool = use_og_country_codes
        self.country_names_fix_dict: dict = country_names_fix_dict

        if self.country_names_list is None:
            self.country_names_list: list = self.get_country_names_list()
        if self.eu_country_names_list is None:
            from setup_module.imports import eu_country_names_list
            self.eu_country_names_list = eu_country_names_list

        if self.use_saved_country_codes is None and self.use_eu_country_codes is None:
            self.use_saved_country_codes = False
            self.use_eu_country_codes = True
        elif self.use_saved_country_codes is None and self.use_eu_country_codes is not None:
            self.use_saved_country_codes = False
        elif self.use_saved_country_codes is not None and self.use_eu_country_codes is None:
            self.use_eu_country_codes = False
        elif self.use_saved_country_codes is not None and self.use_eu_country_codes is not None:
            if len(self.country_names_list) > len(self.eu_country_names_list):
                self.use_saved_country_codes = True
                self.use_eu_country_codes = False
            else:
                self.use_saved_country_codes = False
                self.use_eu_country_codes = True
        if self.use_og_country_codes is None:
            self.use_og_country_codes = False

        if self.country_names_fix_dict is None:
            from setup_module.imports import country_names_fix_dict # type:ignore # isort:skip # fmt:skip # noqa # nopep8
            self.country_names_fix_dict = country_names_fix_dict
        self.df_all_codes = self.fix_df_all_codes()

        self.country_codes_3_dict = self.get_country_codes_3_dict()
        print('='*20)
        print(f'use_saved_country_codes is set to {self.use_saved_country_codes}')
        print(f'use_eu_country_codes is set to {self.use_eu_country_codes}')
        print('='*20)

    def get_og_df_all_codes(self) -> pd.DataFrame:
        self.df_all_codes_path = f'{fb_found_data_path}All ISO Codes.pkl'
        try:
            self.df_all_codes = pd.read_pickle(self.df_all_codes_path)
        except FileNotFoundError:
            self.df_all_codes = pd.read_csv('https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv')
        self.df_all_codes.to_csv(f'{fb_found_data_path}All ISO Codes_OG.csv', index=False)
        self.df_all_codes.to_pickle(f'{fb_found_data_path}All ISO Codes_OG.pkl')
        self.df_all_codes['official_name_en'] = self.df_all_codes['official_name_en'].replace(self.country_names_fix_dict)
        return self.df_all_codes.sort_values(by='official_name_en', ascending=True).reset_index(drop=True)

    def fix_taiwan(self) -> pd.DataFrame:
        self.taiwan_dict: dict = {
            'official_name_en': 'Taiwan',
            'official_name_fr': 'Taïwan',
            'official_name_es': 'Taiwán',
            'official_name_ru': 'Тайвань',
            'official_name_zh': '台湾',
            'official_name_ar': 'تايوان',
            'ISO4217-currency_numeric_code': 901,
            'Global Code': True,
            'ISO4217-currency_alphabetic_code': 'TWD',
            'ISO4217-currency_name': 'New Taiwan dollar',
        }
        taiwan_mask = (self.df_all_codes['ISO3166-1-Alpha-3'].str.strip() == 'TWN') & (self.df_all_codes['ISO3166-1-Alpha-2'].str.strip() == 'TW') & (self.df_all_codes['CLDR display name'].str.strip() == 'Taiwan')
        for key, value in self.taiwan_dict.items():
            self.df_all_codes.loc[taiwan_mask, key] = value
        return self.df_all_codes.sort_values(by='official_name_en', ascending=True).reset_index(drop=True)

    def fix_df_all_codes(self) -> pd.DataFrame:
        self.get_og_df_all_codes()
        self.fix_taiwan()
        return self.df_all_codes

    def fix_country_names_list(self, df: pd.DataFrame = None) -> list:
        from setup_module.imports import fb_available_country_codes
        self.df_all_codes = self.fix_df_all_codes()
        temp_country_names_list = sorted(
            list(
                set(
                    self.df_all_codes['official_name_en'].loc[
                        self.df_all_codes['ISO3166-1-Alpha-2'].isin(
                            fb_available_country_codes,
                        )
                    ].values.tolist(),
                ),
            ),
        )
        if df is None and self.country_names_list is None:
            self.country_names_list = temp_country_names_list
        elif df is None and self.country_names_list is not None:
            if len(temp_country_names_list) < len(self.country_names_list):
                self.country_names_list = sorted(
                    list(set(self.country_names_list)),
                )
            else:
                self.country_names_list = sorted(
                    list(
                        set(
                            [country for country in temp_country_names_list if country in self.country_names_list],
                        ),
                    ),
                )
        elif df is not None and self.country_names_list is None:
            self.country_names_list = sorted(
                list(
                    set(
                        [
                            country
                            for country in df['country'].values.tolist()
                            for country_name, country_code_3 in self.country_codes_3_dict.items()
                            if country.strip().lower() == country_name.strip().lower() and country_code_3.strip().lower() in df['country_code_3'].astype(str).str.strip().str.lower().values.tolist()
                        ],
                    ),
                ),
            )
        elif df is not None and self.country_names_list is not None:
            self.country_names_list = sorted(
                list(
                    set(
                        [
                            country
                            for country in self.country_names_list
                            for country_name, country_code_3 in self.country_codes_3_dict.items()
                            if country.strip().lower() == country_name.strip().lower() and country_code_3.strip().lower() in df['country_code_3'].astype(str).str.strip().str.lower().values.tolist()
                        ],
                    ),
                ),
            )
            with open(f'{data_dir}country_names_list.txt', 'w') as f:
                f.write('\n'.join(self.country_names_list))

        return self.country_names_list

    def get_country_names_list(self, df: pd.DataFrame = None) -> list:
        # if not self.use_saved_country_codes and not self.use_eu_country_codes and not self.use_og_country_codes:
        #     print('Using all_country_names_list.')
        #     self.country_names_list: list = self.fix_country_names_list(df)
        # elif self.use_saved_country_codes and not self.use_eu_country_codes and not self.use_og_country_codes:
        #     print('Using saved country_names_list.')
        #     with open(f'{data_dir}country_names_list.txt', 'r') as f:
        #         country_names_list = f.read().splitlines()
        #     self.country_names_list: list = sorted(country_names_list)
        # elif not self.use_saved_country_codes and not self.use_eu_country_codes and self.use_og_country_codes:
        #     print('Using og_country_names_list.')
        #     from setup_module.imports import og_country_names_list # type:ignore # isort:skip # fmt:skip # noqa # nopep8
        #     self.country_names_list: list = sorted(og_country_names_list)
        # elif not self.use_saved_country_codes and self.use_eu_country_codes and not self.use_og_country_codes:
        #     print('Using eu_country_names_list.')
        # from setup_module.imports import eu_country_names_list
        # self.country_names_list: list = sorted(eu_country_names_list)
        self.all_country_names_list: list = self.fix_country_names_list(df)
        self.country_names_list: list = sorted(self.all_country_names_list)

        with open(f'{data_dir}country_names_list.txt', 'w') as f:
            f.write('\n'.join(self.country_names_list))
        with open(f'{data_dir}all_country_names_list.txt', 'w') as f:
            f.write('\n'.join(self.all_country_names_list))

        if os.path.exists(f'{df_save_dir}df_jobs_for_analysis.pkl') and os.path.isfile(f'{df_save_dir}df_jobs_for_analysis.pkl'):
            with open(f'{data_dir}df_jobs_for_analysis_len.txt', 'r') as f:
                df_jobs_len = int(f.read())
            df_jobs = pd.read_pickle(f'{df_save_dir}df_jobs_for_analysis.pkl')
            assert len(df_jobs) == df_jobs_len, f'DATAFRAME MISSING DATA! DF SHOULD BE OF LENGTH {df_jobs_len} BUT IS OF LENGTH {len(df_jobs)}'
            print(f'Dataframe for analysis loaded with shape: {df_jobs.shape}')

            self.country_names_list = sorted(
                list(
                    set(
                        df_jobs['origin_country'].unique().tolist() + df_jobs['eu_reach_country'].unique().tolist(),
                    ),
                ),
            )

            with open(f'{data_dir}country_names_list.txt', 'w') as f:
                f.write('\n'.join(self.country_names_list))

        return self.country_names_list

    # Secondary functions to get country codes dicts and lists
    def get_country_codes_num_dict(self) -> dict:
        self.country_codes_num_dict = {}
        for country in self.country_names_list:
            if country in self.df_all_codes['official_name_en'].tolist():
                self.country_codes_num_dict[country] = self.df_all_codes[self.df_all_codes['official_name_en'].str.strip() == country.strip()]['ISO3166-1-numeric'].values[0]
            else:
                print(f'{country} not found in ISO 3166 Country Codes.csv column official_name_en')
        with open(f'{data_dir}country_codes_num_dict.json', 'w') as f:
            json.dump(self.country_codes_num_dict, f, indent=4)
        return self.country_codes_num_dict

    def get_country_codes_num_dict_reversed(self) -> dict:
        self.country_codes_num_dict = self.get_country_codes_num_dict()
        self.country_codes_num_dict_reversed = {v: k.strip() for k, v in self.country_codes_num_dict.items()}
        with open(f'{data_dir}country_codes_num_dict_reversed.json', 'w') as f:
            json.dump(self.country_codes_num_dict_reversed, f, indent=4)
        return self.country_codes_num_dict_reversed

    def get_country_codes_dict(self) -> dict:
        self.country_codes_dict = {}
        for country in self.country_names_list:
            if country in self.df_all_codes['official_name_en'].tolist():
                self.country_codes_dict[country] = self.df_all_codes[self.df_all_codes['official_name_en'].str.strip() == country.strip()]['ISO3166-1-Alpha-2'].values[0].strip()
            else:
                print(f'{country} not found in ISO 3166 Country Codes.csv')
        with open(f'{data_dir}country_codes_dict.json', 'w') as f:
            json.dump(self.country_codes_dict, f, indent=4)
        return self.country_codes_dict

    def get_country_codes_dict_reversed(self) -> dict:
        self.country_codes_dict = self.get_country_codes_dict()
        self.country_codes_dict_reversed = {v.strip(): k.strip() for k, v in self.country_codes_dict.items()}
        with open(f'{data_dir}country_codes_dict_reversed.json', 'w') as f:
            json.dump(self.country_codes_dict_reversed, f, indent=4)
        return self.country_codes_dict_reversed

    def get_country_codes_list(self) -> list:
        self.country_codes_dict = self.get_country_codes_dict()
        self.country_codes_list = list(self.country_codes_dict.values())
        with open(f'{data_dir}country_codes_list.txt', 'w') as f:
            f.write('\n'.join(self.country_codes_list))
        return self.country_codes_list

    def get_country_codes_3_dict(self) -> dict:
        # Make country_codes_3_dict from df_all_codes where name is key and country_code is value if the country is in country_names_list
        self.country_codes_3_dict = {}
        for country in self.country_names_list:
            if country in self.df_all_codes['official_name_en'].tolist():
                self.country_codes_3_dict[country] = self.df_all_codes[
                    self.df_all_codes['official_name_en'].str.strip(
                    ) == country.strip()
                ]['ISO3166-1-Alpha-3'].values[0].strip()
            else:
                print(f'{country} not found in ISO 3166 Country Codes.csv')
        with open(f'{data_dir}country_codes_3_dict.json', 'w') as f:
            json.dump(self.country_codes_3_dict, f, indent=4)
        return self.country_codes_3_dict

    def get_country_codes_3_dict_reversed(self) -> dict:
        self.country_codes_3_dict = self.get_country_codes_3_dict()
        self.country_codes_3_dict_reversed = {v.strip(): k.strip() for k, v in self.country_codes_3_dict.items()}
        with open(f'{data_dir}country_codes_3_dict_reversed.json', 'w') as f:
            json.dump(self.country_codes_3_dict_reversed, f, indent=4)
        return self.country_codes_3_dict_reversed

    def get_country_codes_3_list(self) -> list:
        self.country_codes_3_dict = self.get_country_codes_3_dict()
        self.country_codes_3_list = list(self.country_codes_3_dict.values())
        with open(f'{data_dir}country_codes_3_list.txt', 'w') as f:
            f.write('\n'.join(self.country_codes_3_list))
        return self.country_codes_3_list

    def get_country_codes_2_and_3_dict(self) -> dict:
        self.country_codes_dict = self.get_country_codes_dict()
        self.country_codes_3_dict = self.get_country_codes_3_dict()
        self.country_codes_2_and_3_dict = {}

        for country_name_2, country_code_2 in self.country_codes_dict.items():
            for country_name_3, country_code_3 in self.country_codes_3_dict.items():
                if country_name_2 == country_name_3:
                    self.country_codes_2_and_3_dict[country_name_2] = [country_code_2.strip(), country_code_3.strip()]
        with open(f'{data_dir}country_codes_2_and_3_dict.json', 'w') as f:
            json.dump(self.country_codes_2_and_3_dict, f, indent=4)
        return self.country_codes_2_and_3_dict

    def get_country_codes_2_and_3_num_dict(self) -> dict:
        self.country_codes_2_and_3_dict = self.get_country_codes_2_and_3_dict()
        self.country_codes_num_dict = self.get_country_codes_num_dict()
        self.country_codes_2_and_3_num_dict = {}

        for country_name, country_codes_2_and_3 in self.country_codes_2_and_3_dict.items():
            for country_name_num, country_code_num in self.country_codes_num_dict.items():
                if country_name == country_name_num:
                    self.country_codes_2_and_3_num_dict[country_name] = [
                        country_codes_2_and_3[0].strip(), country_codes_2_and_3[-1].strip(), country_code_num,
                    ]
        with open(f'{data_dir}country_codes_2_and_3_num_dict.json', 'w') as f:
            json.dump(self.country_codes_2_and_3_num_dict, f, indent=4)
        return self.country_codes_2_and_3_num_dict

    def get_currency_codes_dict(self) -> dict:
        self.currency_codes_dict = {
            currency_name.strip(): currency_code.strip()
            for country_name, currency_name, currency_code in self.df_all_codes[['official_name_en', 'ISO4217-currency_name', 'ISO4217-currency_alphabetic_code']].values
            if isinstance(currency_name, str)
            and isinstance(currency_code, str)
            and currency_name.strip() != 'nan'
            and country_name.strip() in self.country_names_list
        }
        with open(f'{data_dir}currency_codes_dict.json', 'w') as f:
            json.dump(self.currency_codes_dict, f, indent=4)
        return self.currency_codes_dict

    def get_currency_codes_dict_reversed(self) -> dict:
        self.currency_codes_dict = self.get_currency_codes_dict()
        self.currency_codes_dict_reversed = {v.strip(): k.strip() for k, v in self.currency_codes_dict.items()}
        with open(f'{data_dir}currency_codes_dict_reversed.json', 'w') as f:
            json.dump(self.currency_codes_dict_reversed, f, indent=4)
        return self.currency_codes_dict_reversed

    def get_country_currency_codes_dict(self) -> dict:
        self.country_currency_codes_dict = {
            country_name.strip(): currency_code.strip()
            for country_name, currency_code in self.df_all_codes[['official_name_en', 'ISO4217-currency_alphabetic_code']].values
            if isinstance(country_name, str)
            and isinstance(currency_code, str)
            and currency_code.strip() != 'nan'
            and country_name in self.country_names_list
        }
        with open(f'{data_dir}country_currency_codes_dict.json', 'w') as f:
            json.dump(self.country_currency_codes_dict, f, indent=4)
        return self.country_currency_codes_dict

    def get_country_currency_codes_dict_reversed(self) -> dict:
        self.country_currency_codes_dict = self.get_country_currency_codes_dict()
        self.country_currency_codes_dict_reversed = {v.strip(): k.strip() for k, v in self.country_currency_codes_dict.items()}
        with open(f'{data_dir}country_currency_codes_dict_reversed.json', 'w') as f:
            json.dump(self.country_currency_codes_dict_reversed, f, indent=4)
        return self.country_currency_codes_dict_reversed

    def get_language_codes_dict_with_hyphen(self) -> dict:
        self.language_codes_dict_with_hyphen = {
            official_name_en: languages.split(',')
            for official_name_en, languages in self.df_all_codes[['official_name_en', 'Languages']].values
            if isinstance(languages, str) and official_name_en in self.country_names_list
        }
        with open(f'{data_dir}language_codes_dict_with_hyphen.json', 'w') as f:
            json.dump(self.language_codes_dict_with_hyphen, f, indent=4)
        return self.language_codes_dict_with_hyphen

    def get_language_codes_dict(self) -> dict:
        self.language_codes_dict_with_hyphen = self.get_language_codes_dict_with_hyphen()
        self.language_codes_dict = {}
        for country, languages in self.language_codes_dict_with_hyphen.items():
            self.language_codes_dict[country] = list(
                {
                    language.split('-')[0].strip()
                    for language in languages
                    if isinstance(language, str) and language.strip() != 'nan' and len(language.split('-')[0].strip()) == 2
                },
            )
        with open(f'{data_dir}language_codes_dict.json', 'w') as f:
            json.dump(self.language_codes_dict, f, indent=4)
        return self.language_codes_dict

    def get_language_codes_dict_reversed(self) -> dict:
        self.language_codes_dict = self.get_language_codes_dict()
        # make language_codes_dict_reversed from language_codes_dict where each language code is a key and if that language code appears in multiple countries, the value is a list of countries
        self.language_codes_dict_reversed = {}
        for country, languages in self.language_codes_dict.items():
            for language in languages:
                if isinstance(language, str) and language.strip() != 'nan':
                    if language in self.language_codes_dict_reversed:
                        self.language_codes_dict_reversed[language].append(
                            country.strip(),
                        )
                    else:
                        self.language_codes_dict_reversed[language] = [
                            country.strip(),
                        ]
        # Remove duplicates in self.language_codes_dict_reversed values list
        for key, value in self.language_codes_dict_reversed.items():
            self.language_codes_dict_reversed[key] = list(set(value))
        with open(f'{data_dir}language_codes_dict_reversed.json', 'w') as f:
            json.dump(self.language_codes_dict_reversed, f, indent=4)
        return self.language_codes_dict_reversed

    def get_primary_language_codes_dict(self) -> dict:
        self.language_codes_dict = self.get_language_codes_dict()
        self.primary_language_codes_dict = {}
        for key, value in self.language_codes_dict.items():
            self.primary_language_codes_dict[key] = value[0].split('-')[0].strip()
        with open(f'{data_dir}primary_language_codes_dict.json', 'w') as f:
            json.dump(self.primary_language_codes_dict, f, indent=4)
        return self.primary_language_codes_dict

    def get_primary_language_codes_dict_reversed(self) -> dict:
        self.primary_language_codes_dict = self.get_primary_language_codes_dict()
        self.primary_language_codes_dict_reversed = {v: k for k, v in self.primary_language_codes_dict.items()}
        with open(f'{data_dir}primary_language_codes_dict_reversed.json', 'w') as f:
            json.dump(self.primary_language_codes_dict_reversed, f, indent=4)
        return self.primary_language_codes_dict_reversed

    def get_languages_only_dict(self) -> dict:
        self.df_language_codes = self.get_df_language_codes()
        self.languages_only_dict = dict(zip(self.df_language_codes['language_code'], self.df_language_codes['language_name']))
        with open(f'{data_dir}languages_only_dict.json', 'w') as f:
            json.dump(self.languages_only_dict, f, indent=4)
        return self.languages_only_dict

    def get_languages_only_dict_reversed(self) -> dict:
        self.languages_only_dict = self.get_languages_only_dict()
        self.languages_only_dict_reversed = {v: k for k, v in self.languages_only_dict.items()}
        with open(f'{data_dir}languages_only_dict_reversed.json', 'w') as f:
            json.dump(self.languages_only_dict_reversed, f, indent=4)
        return self.languages_only_dict_reversed

    def get_country_languages_dict(self) -> dict:
        self.country_languages_dict = defaultdict(list)
        self.language_codes_dict = self.get_language_codes_dict()
        self.languages_only_dict = self.get_languages_only_dict()
        for (country, language_codes_list), (language_code, language_name) in itertools.product(self.language_codes_dict.items(), self.languages_only_dict.items()):
            for lang_code in language_codes_list:
                if lang_code == language_code:
                    self.country_languages_dict[country].append(language_name)
        self.country_languages_dict = dict(self.country_languages_dict)
        with open(f'{data_dir}country_languages_dict.json', 'w') as f:
            json.dump(self.country_languages_dict, f, indent=4)
        return self.country_languages_dict

    def get_df_all_codes(self) -> pd.DataFrame:
        self.df_all_codes.to_csv(f'{fb_found_data_path}All ISO Codes.csv', index=False)
        self.df_all_codes.to_pickle(f'{fb_found_data_path}All ISO Codes.pkl')
        return self.df_all_codes.sort_values(by='official_name_en', ascending=True)

    def make_df_codes_table(self) -> None:
        self.country_codes_2_and_3_num_dict = self.get_country_codes_2_and_3_num_dict()
        self.df_codes_table = pd.DataFrame.from_dict(
            self.country_codes_2_and_3_num_dict,
            orient='index',
            columns=[
                '2-letter Code: ISO3166-1-Alpha-2', '3-letter Code: ISO3166-1-Alpha-3', 'Numeric Code: ISO3166-1-numeric',
            ],
        )\
            .reset_index(drop=False)\
                .rename(columns={'index': 'Country'})
        self.df_codes_table.insert(0, 'Count ID', range(1, 1 + len(self.df_codes_table)))
        self.df_codes_table['Country'] = self.df_codes_table['Country']\
            .replace(country_names_fix_dict)
        return self.df_codes_table

    def print_info(self) -> typing.Callable:
        print('='*20)
        print(f'Size of df_all_codes: {self.df_all_codes.shape}')
        print(f'Columns in df_all_codes:\n{self.df_all_codes.columns.tolist()}', end='\n')
        print(f'df_all_codes head:\n{self.df_all_codes.head(3)}', end='\n')
        print('-'*20, end='\n')
        print(f'Number of Countries: {len(self.country_names_list)}')
        print(f'Country List: {self.country_names_list}')
        with contextlib.suppress(NameError, AttributeError):
            print(f'All Facebook Country List (len={len(self.all_country_names_list)}): {self.all_country_names_list}')
        print(f'Country Codes:\n{self.country_codes_2_and_3_num_dict}', end='\n')
        print('-'*20, end='\n')
        print(f'Number of Languages: {len(self.language_codes_dict)}')
        print(f'Language Codes:\n{self.country_languages_dict}', end='\n')
        print('-'*20, end='\n')
        print(f'Number of convert_to_currency_code: {len(self.currency_codes_dict)}')
        print(f'Currency Codes:\n{self.country_currency_codes_dict}', end='\n')
        print('='*20)

    def get_all_code_data(self, print_enabled: bool = True, return_enabled: bool = True) -> tuple[pd.DataFrame, list, dict, dict, dict, dict, dict, dict, dict, dict, dict]:
        self.df_all_codes = self.get_df_all_codes()
        self.country_codes_list = self.get_country_codes_list()
        self.country_codes_dict = self.get_country_codes_dict()
        self.country_codes_dict_reversed = self.get_country_codes_dict_reversed()
        self.country_codes_3_list = self.get_country_codes_3_list()
        self.country_codes_3_dict = self.get_country_codes_3_dict()
        self.country_codes_3_dict_reversed = self.get_country_codes_3_dict_reversed()
        self.country_codes_2_and_3_dict = self.get_country_codes_2_and_3_dict()
        self.country_codes_num_dict = self.get_country_codes_num_dict()
        self.country_codes_num_dict_reversed = self.get_country_codes_num_dict_reversed()
        self.country_codes_2_and_3_num_dict = self.get_country_codes_2_and_3_num_dict()
        self.currency_codes_dict = self.get_currency_codes_dict()
        self.currency_codes_dict_reversed = self.get_currency_codes_dict_reversed()
        self.country_currency_codes_dict = self.get_country_currency_codes_dict()
        self.country_currency_codes_dict_reversed = self.get_country_currency_codes_dict_reversed()
        self.language_codes_dict_with_hyphen = self.get_language_codes_dict_with_hyphen()
        self.language_codes_dict = self.get_language_codes_dict()
        self.language_codes_dict_reversed = self.get_language_codes_dict_reversed()
        self.primary_language_codes_dict = self.get_primary_language_codes_dict()
        self.primary_language_codes_dict_reversed = self.get_primary_language_codes_dict_reversed()
        self.languages_only_dict = self.get_languages_only_dict()
        self.languages_only_dict_reversed = self.get_languages_only_dict_reversed()
        self.country_languages_dict = self.get_country_languages_dict()
        self.save_df_codes_table()
        if print_enabled:
            self.print_info()

        if return_enabled:
            return (
                self.df_all_codes,
                self.country_names_list,
                self.country_codes_list,
                self.country_codes_dict,
                self.country_codes_dict_reversed,
                self.country_codes_3_list,
                self.country_codes_3_dict,
                self.country_codes_3_dict_reversed,
                self.country_codes_2_and_3_dict,
                self.country_codes_num_dict,
                self.country_codes_num_dict_reversed,
                self.country_codes_2_and_3_num_dict,
                self.currency_codes_dict,
                self.currency_codes_dict_reversed,
                self.country_currency_codes_dict,
                self.country_currency_codes_dict_reversed,
                self.language_codes_dict_with_hyphen,
                self.language_codes_dict,
                self.language_codes_dict_reversed,
                self.primary_language_codes_dict,
                self.primary_language_codes_dict_reversed,
                self.languages_only_dict,
                self.languages_only_dict_reversed,
                self.country_languages_dict,
            )

    # Child class methods
    def fix_country_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[
            df['country_code_3'].astype(str).str.strip().str.lower().isin(
                [
                    code_3.strip().lower()
                    for code_3 in self.country_codes_3_dict.values()
                ],
            )
        ].sort_values(by='country', ascending=True).reset_index(drop=True)
        for country, country_code_3 in self.country_codes_3_dict.items():
            df.loc[df['country_code_3'].astype(str).str.strip().str.lower() == country_code_3.strip().lower(), 'country'] = country
        return df.loc[
            df['country'].astype(str).str.strip().str.lower().isin(
                [
                    country.strip().lower()
                    for country in self.country_names_list
                ],
            )
        ].sort_values(by='country', ascending=True).reset_index(drop=True)

    def add_country_codes_and_country_num_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fix_country_names(df=df)
        all_codes = self.get_all_code_data(print_enabled=False)
        df['country_code'] = df['country'].map(self.country_codes_dict)
        df['country_num'] = df['country'].map(self.country_codes_num_dict)
        return df

# %%
# %%
# Get Currency Converter
# https://www.thepythoncode.com/article/make-a-currency-converter-in-python
# https://manage.exchangeratesapi.io/dashboard
# date format: YYYY-MM-DD, e.g. 2023-05-31. 'ad_delivery_stop_time' is in the same format
class CurrencyConverter:

    env_path = Path(code_dir).joinpath('.envrc')
    load_dotenv(dotenv_path=env_path)

    def __init__(
        self,
        date: str = None,
        base_currency_code: str = None,
        convert_to_currency_code: str = None,
        problematic_currency: str = None,
        base_url: str = None,
        amount: float = None,
        exchange_paid_api: bool = None,
        access_token: str = None,
        api_version: str = None,
    ) -> None:
        self.__access_token: str = os.environ.get('EXCHANGE_RATES_API_KEY').replace('"', '').replace("'", '')
        self.api_version: str = os.environ.get('EXCHANGE_RATES_API_VERSION')
        self._date: str = date
        self._base_currency_code: str = base_currency_code
        self._convert_to_currency_code: Union[None, str, list] = convert_to_currency_code
        self._problematic_currency: str = 'USD'
        self._base_url: str = base_url
        self.amount: float = 1.000000 if amount is None else amount
        self.exchange_paid_api: bool = exchange_paid_api

        if self.exchange_paid_api is None:
            self.exchange_paid_api = False

    # Private property
    @property
    def date(self) -> str:
        if self._date is None:
            self._date = datetime.datetime.now().strftime('%Y-%m-%d')
        return self._date

    @date.setter
    def date(self, date: str) -> str:
        self._date = date

    @property
    def access_token(self) -> str:
        one_fourth = len(self.__access_token) // 5
        return f'{self.__access_token[:one_fourth]}{"*" * (len(self.__access_token) - (one_fourth*2))}{self.__access_token[-one_fourth:]}'

    @access_token.setter
    def access_token(self, access_token: str) -> str:
        self.__access_token = access_token

    @property
    def __base_currency_code(self) -> str:
        if (self._base_currency_code is None or self._base_currency_code == self._problematic_currency) and self.exchange_paid_api is False:
            return 'EUR'
        return self._base_currency_code

    @property
    def base_currency_code(self) -> str:
        if (self._base_currency_code is None or self._base_currency_code != self._problematic_currency) and self.exchange_paid_api is False:
            return self.__base_currency_code
        return self._base_currency_code

    @base_currency_code.setter
    def base_currency_code(self, base_currency_code: str) -> str:
        self._base_currency_code = _base_currency_code

    @property
    def convert_to_currency_code(self) -> str:
        if isinstance(self._convert_to_currency_code, list):
            self._convert_to_currency_code = ','.join(
                list(
                    set(
                        code for code in self._convert_to_currency_code if code not in nan_list
                    ),
                ),
            )
        if self._base_currency_code == self._problematic_currency and self.exchange_paid_api is False:
            self._convert_to_currency_code = f'{self._convert_to_currency_code},{self._problematic_currency}'
        else:
            if self._convert_to_currency_code is not None and self._convert_to_currency_code != self.__base_currency_code:
                self._convert_to_currency_code = f'{self._convert_to_currency_code},{self.__base_currency_code}'
            elif self._convert_to_currency_code is None or self._convert_to_currency_code == self.__base_currency_code:
                self._convert_to_currency_code = self.__base_currency_code
        return self._convert_to_currency_code

    @convert_to_currency_code.setter
    def convert_to_currency_code(self, convert_to_currency_code) -> str:
        self._convert_to_currency_code = convert_to_currency_code

    @property
    def url(self) -> str:
        if self._base_url is None:
            self._base_url = self._base_url = f'api.exchangeratesapi.io/{self.api_version}/latest?access_key={self.__access_token}'
            if self.exchange_paid_api:
                self.url_prefix = 'https://'
            else:
                self.url_prefix = 'http://'
        if self.date is not None:
            self._url = f'{self.url_prefix}{self._base_url.replace("latest", str(self.date))}'

        return self._url

    @url.setter
    def url(self, url: str) -> str:
        self._url = url

    @property
    def headers(self) -> dict:
        return {'Authorization': f'Bearer {self.__access_token}'}

    @headers.setter
    def headers(self, headers: dict) -> dict:
        self.headers = headers

    @property
    def params(self) -> dict:
        self._params: dict = {
            'access_key': self.__access_token,
        }
        if self.__base_currency_code is not None:
            self._params['base'] = self.__base_currency_code
        if self.convert_to_currency_code is not None:
            self._params['symbols'] = ','.join(
                [
                    symbol.strip()
                    for symbol in list(set(self.convert_to_currency_code.strip().split(',')))
                ],
            )
        return self._params

    @params.setter
    def params(self, params: dict) -> dict:
        self._params = params

    def get_request(self, url: Optional[str] = None) -> requests.models.Response:
        if url is None:
            url = self.url
        return requests.get(url=url, headers=self.headers, params=self.params)

    def get_status_code(self) -> int:
        return self.get_request().status_code

    def get_success(self) -> bool:
        return self.get_request().json()['success']

    def get_headers(self) -> dict:
        return self.get_request().headers

    def get_json(self) -> dict:
        if self.get_status_code() == 200 and self.get_success():
            return self.get_request().json()
        else:
            raise Exception(f'Error: {self.get_status_code()}\nSuccess: {self.get_success()}\nHeaders: {self.get_headers()}')

    def get_exchange_rates_json(self) -> dict:
        return self.get_json()['rates']

    def change_base_currency_eu_to_usd(func: typing.Callable) -> typing.Callable:
        def wrapper(self, *args, **kwargs) -> typing.Any:
            exchange_rates = func(self, *args, **kwargs)
            if exchange_rates is not None:
                exchange_rates = {k: v for k, v in exchange_rates.items() if v}
                if self.base_currency_code == self._problematic_currency and self.exchange_paid_api is False:
                    # one_usd_to_eur = 1 (EUR) / 1.149155 (USD) = 0.869565
                    # one_usd_to_eur = 1 (USD) * 0.869565 (EUR) = 0.869565
                    one_usd_to_eur = exchange_rates[self.__base_currency_code] / exchange_rates[self.base_currency_code]
                    exchange_rates = {
                        currency: round(one_usd_to_eur * rate, 6)
                        for currency, rate in exchange_rates.items()
                    }
                    if self.__base_currency_code in self._convert_to_currency_code:
                        exchange_rates[self.__base_currency_code] = round(one_usd_to_eur, 6)
                if self.base_currency_code not in exchange_rates.keys():
                    exchange_rates[self.__base_currency_code] = 1.000000
                return exchange_rates
            else:
                return func(self, *args, **kwargs)
        return wrapper

    @change_base_currency_eu_to_usd
    def get_exchange_rates(self) -> dict:
        return self.get_exchange_rates_json()

    def calculate_currency(self) -> float:
        self.exchange_rates = self.get_exchange_rates()
        if self.exchange_rates is not None:
            for currency in convert_to_currency_code.split(','):
                if currency in self.exchange_rates.keys() and symbol != self.base_currency_code:
                    return round(self.amount * self.exchange_rates[symbol], 2)

# %%
# Make Base Class for GDPData, GiniCoefficient, and PopulationData
class WBBaseClass(AllCodes):

    def __init__(
        self,
        series_search: str = None,
        series_name: str = None,
        series_id: str = None,
        series_directory: str = None,
        start_year: int = None,
        end_year: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.series_search: str = series_search
        self._series_name: str = series_name
        self._series_id: str = series_id
        self._series_directory: str = series_directory
        self._start_year: str = start_year
        self._end_year: str = end_year

        self._start_year = 2018 if self._start_year is None else self._start_year
        self._end_year = 2023 if self._end_year is None else self._end_year

    @property
    def start_year(self) -> str:
        if isinstance(self._start_year, str):
            return int(self._start_year)
        else:
            return self._start_year
    @start_year.setter
    def start_year(self, start_year: str) -> str:
        self._start_year = start_year

    @property
    def end_year(self) -> str:
        if isinstance(self._end_year, str):
            return int(self._end_year)
        else:
            return self._end_year
    @end_year.setter
    def end_year(self, end_year: str) -> str:
        self._end_year = end_year

    @property
    def series_name(self) -> str:
        if self._series_name is None:
            if self._series_id is not None:
                return wb.series.info(self._series_id).items[0]['value']
            elif self._series_id is None:
                if self.series_search is None:
                    self.series_search = str(input('No series search term is set. Please input a series search term.'))
                series_names = wb.series.info(q=self.series_search.replace("'", "").strip()).items
        elif self._series_name is not None:
            return self._series_name
        self._series_name = str(
            input(f'No series name is set. Please input a series name from the following:\
            \n{[item["value"] for item in series_names]}'),
        )
        return self._series_name

    @series_name.setter
    def series_name(self, series_name: str) -> None:
        self._series_name = series_name

    @property
    def series_id(self) -> str:
        if self._series_id is None:
            if self.series_name is not None:
                return wb.series.info(q=self.series_name.replace("'", "").strip()).items[0]['id']
            elif self.series_search is not None:
                return wb.series.info(q=self.series_search.replace("'", "").strip()).items[0]['id']
            else:
                raise Exception('No series name or search term is set.')
        elif self._series_id is not None:
            return self._series_id

    @series_id.setter
    def series_id(self, series_id: str) -> None:
        self._series_id = series_id

    @property
    def series_directory(self) -> str:
        if self._series_directory is None:
            if re.search(r'(?i)gdp', self.series_name):
                return 'GDP'
            elif re.search(r'(?i)gini', self.series_name):
                return 'GINI'
            elif re.search(r'(?i)population', self.series_name):
                return 'POPULATION'
            elif re.search(r'(?i)labor', self.series_name):
                return 'LABOR'
            elif re.search(r'(?i)unemployment', self.series_name):
                return 'UNEMPLOYMENT'
            elif re.search(r'(?i)internet', self.series_name):
                return 'INTERNET'
            else:
                raise Exception('Series name does not contain GDP, Gini, or Population data.')
        return self._series_directory

    @series_directory.setter
    def series_directory(self, series_directory: str) -> None:
        self._series_directory = series_directory

    def get_og_df(self) -> pd.DataFrame:
        self.df = wb.data.DataFrame(series=self.series_id, time=range(self.start_year, self.end_year+1), labels=True, skipBlanks=False)
        self.df.to_csv(f'{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_directory}_OG.csv')
        self.df.to_pickle(f'{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_directory}_OG.pkl')
        return self.df

    def clean_og_df(self) -> pd.DataFrame:
        self.df = self.get_og_df()
        self.df = self.df.reset_index(drop=False).rename(columns={'Country': 'country', 'economy': 'country_code_3'})
        self.df = pd.concat(
            [self.df, self.df.loc[self.df['country'] == 'China'].replace({'China': 'Taiwan', 'CHN': 'TWN'})],
            ignore_index=True,
        )
        self.df = self.df.drop_duplicates(
            subset=['country', 'country_code_3'], keep='first',
        ).dropna(
            how='all',
        ).sort_values(by='country', ascending=True).reset_index(drop=True)
        return self.df

    def z_score_and_trichotimize(self, df, col: str) -> pd.DataFrame:
        df[f'z_{col}'] = scale(df[col])
        df[f'{col}_categorical_threshold'] = pd.qcut(df[col], q=3)
        df[f'z_{col}_categorical_threshold'] = pd.qcut(df[f'z_{col}'], q=3)
        df[f'{col}_categorical'] = pd.qcut(df[col], q=3, labels=['low', 'medium', 'high'])
        df[f'{col}_categorical_num'] = pd.qcut(df[col], q=3, labels=[-1, 0, 1])
        return df

    def update_country_names_list_and_get_all_code_data(
        self, df: pd.DataFrame, use_saved_country_codes: bool, use_eu_country_codes: bool,
    ) -> list:
        country_names_list = AllCodes(
            use_saved_country_codes=use_saved_country_codes, use_eu_country_codes=use_eu_country_codes,
        ).get_country_names_list(df=self.df)
        super().__init__(
            country_names_list=country_names_list, use_saved_country_codes=use_saved_country_codes, use_eu_country_codes=use_eu_country_codes,
        )
        self.get_all_code_data(print_enabled=False, return_enabled=False)
        return country_names_list

# %%
# Get GDP
# https://data.oecd.org/gdp/gross-domestic-product-gdp.htm
# https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
class GDPData(WBBaseClass):

    def __init__(
        self,
        series_directory: str = None,
        base_currency_code: str = None,
        convert_to_currency_code: str = None,
        exchange_paid_api: bool = None,
        use_saved_country_codes: bool = None,
        use_eu_country_codes: bool = None,
        **kwargs,
    ) -> None:
        self._series_directory: str = series_directory
        self.base_currency_code: str = base_currency_code
        self.convert_to_currency_code: str = convert_to_currency_code
        self.exchange_paid_api: bool = exchange_paid_api
        self.use_saved_country_codes: bool = use_saved_country_codes
        self.use_eu_country_codes: bool = use_eu_country_codes

        if self._series_directory is None:
            self._series_directory: str = 'GDP'
        if self.base_currency_code is None or 'US' in self.series_name:
            self.base_currency_code: str = 'USD'
        if self.convert_to_currency_code is None or 'EU' in self.series_name:
            self.convert_to_currency_code: str = 'EUR'
        if self.exchange_paid_api is None:
            self.exchange_paid_api: bool = False
        if self.use_saved_country_codes is None:
            self.use_saved_country_codes: bool = False
        if self.use_eu_country_codes is None:
            self.use_eu_country_codes: bool = True

        # Call to WBBaseClass without passing series-related parameters explicitly
        super().__init__(
            series_directory=self._series_directory,
            use_saved_country_codes=self.use_saved_country_codes,
            use_eu_country_codes=self.use_eu_country_codes,
            **kwargs,
        )

        self.year_rename_dict, self.available_years, self.latest_year, self.df_gdp = self.clean_og_df_gdp()
        self.country_names_list = self.update_country_names_list_and_get_all_code_data(
            df=self.df_gdp, use_saved_country_codes=self.use_saved_country_codes, use_eu_country_codes=self.use_eu_country_codes,
        )

    def clean_og_df_gdp(self) -> pd.DataFrame:
        self.df_gdp = self.clean_og_df()
        self.year_rename_dict = {
            f'{col}': f'{col.split("YR")[1]}_{self.base_currency_code}'
            for col in self.df.columns.tolist()
            if 'YR' in col
        }
        self.available_years = sorted(
            [
                f'{yr.replace(self.base_currency_code, self.convert_to_currency_code)}'
                for yr in list(self.year_rename_dict.values())
            ],
        )
        self.latest_year = int(max(self.available_years, key=lambda x: int(''.join(filter(str.isdigit, x)))).split('_')[0])
        self.df = self.df.reset_index(drop=True).rename(columns=self.year_rename_dict)
        return self.year_rename_dict, self.available_years, self.latest_year, self.df

    def get_df_gdp_no_euro(self) -> pd.DataFrame:
        self.df_gdp = self.add_country_codes_and_country_num_column(df=self.df_gdp)
        return  self.df_gdp.drop(columns=[col for col in self.df_gdp.columns.tolist() if self.convert_to_currency_code in col])

    def eur_to_usd_base_currency(func: typing.Callable) -> typing.Callable:
        def wrapper(self, *args, **kwargs) -> typing.Callable:
            exchange_rates = func(self, *args, **kwargs)
            if self.base_currency_code == self.convert_to_currency_code:
                exchange_rates = {
                    year: {
                        currency: 1 / rate if currency == self.convert_to_currency_code else rate for currency, rate in rates.items()
                    }
                    for year, rates in exchange_rates.items()
                }
            return exchange_rates
        return wrapper

    @eur_to_usd_base_currency
    def get_gdp_exchange_rates(self, base_currency_code: str = None, convert_to_currency_code: str = None, exchange_paid_api: bool = None) -> dict:
        base_currency_code = self.base_currency_code if base_currency_code is None else base_currency_code
        convert_to_currency_code = self.convert_to_currency_code if convert_to_currency_code is None else convert_to_currency_code
        exchange_paid_api = self.exchange_paid_api if exchange_paid_api is None else exchange_paid_api
        self.exchange_rates = {}

        for year in range(self.start_year, self.end_year+1):
            self.exchange_rates[year] = CurrencyConverter(date=f'{year}-12-31', base_currency_code=base_currency_code, convert_to_currency_code=convert_to_currency_code, exchange_paid_api=exchange_paid_api).get_exchange_rates()
        with open(f'{data_dir}{self.start_year} - {self.end_year} base_currency 1 {self.base_currency_code} = {self.convert_to_currency_code} exchange_rates.json', 'w') as f:
            json.dump(self.exchange_rates, f, indent=4)
        return self.exchange_rates

    def get_df_gdp_euro(self, base_currency_code: str = None, convert_to_currency_code: str = None, exchange_paid_api: bool = None) -> pd.DataFrame():
        base_currency_code = self.base_currency_code if base_currency_code is None else base_currency_code
        convert_to_currency_code = self.convert_to_currency_code if convert_to_currency_code is None else convert_to_currency_code
        exchange_paid_api = self.exchange_paid_api if exchange_paid_api is None else exchange_paid_api

        self.df_gdp = self.get_df_gdp_no_euro()
        self.exchange_rates = self.get_gdp_exchange_rates(base_currency_code=base_currency_code, convert_to_currency_code=convert_to_currency_code, exchange_paid_api=exchange_paid_api)

        for col in self.df_gdp.columns.tolist():
            if self.base_currency_code in col and int(f'{col.split("_USD")[0]}') in self.exchange_rates.keys():
                for idx, row in self.df_gdp.iterrows():
                    # one_usd_to_eur = 1 (EUR) / 1.149155 (USD) = 0.869565
                    # one_usd_to_eur = 1 (USD) * 0.869565 (EUR) = 0.869565
                    self.df_gdp.loc[idx, f'{col.split(f"_{self.base_currency_code}")[0]}_EUR'] = row[col] * self.exchange_rates[int(f'{col.split(f"_{self.base_currency_code}")[0]}')][self.convert_to_currency_code]
        return self.df_gdp

    def get_df_gdp(self, base_currency_code: str = None, convert_to_currency_code: str = None, exchange_paid_api: bool = None) -> pd.DataFrame:
        base_currency_code = self.base_currency_code if base_currency_code is None else base_currency_code
        convert_to_currency_code = self.convert_to_currency_code if convert_to_currency_code is None else convert_to_currency_code
        exchange_paid_api = self.exchange_paid_api if exchange_paid_api is None else exchange_paid_api
        self.df_gdp = self.get_df_gdp_euro(base_currency_code=base_currency_code, convert_to_currency_code=convert_to_currency_code, exchange_paid_api=exchange_paid_api)
        self.df_gdp = self.z_score_and_trichotimize(self.df_gdp, f'{self.latest_year}_{self.convert_to_currency_code}')

        self.df_gdp.to_csv(f'{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_directory}.csv', index=False)
        self.df_gdp.to_pickle(f'{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_directory}.pkl')
        return self.df_gdp

# %%
# Get GINI Coefficient
class GiniCoefficient(WBBaseClass):

    def __init__(
        self,
        series_directory: str = None,
        base_currency_code: str = None,
        convert_to_currency_code: str = None,
        use_saved_country_codes: bool = None,
        use_eu_country_codes: bool = None,
        **kwargs,
    ) -> None:
        self._series_directory: str = series_directory
        self.base_currency_code: str = base_currency_code
        self.convert_to_currency_code: str = convert_to_currency_code
        self.use_saved_country_codes: bool = use_saved_country_codes
        self.use_eu_country_codes: bool = use_eu_country_codes

        if self._series_directory is None:
            self._series_directory: str = 'GINI'
        if self.base_currency_code is None or 'US' in self.series_name:
            self.base_currency_code: str = 'USD'
        if self.convert_to_currency_code is None or 'EU' in self.series_name:
            self.convert_to_currency_code: str = 'EUR'
        if self.use_saved_country_codes is None:
            self.use_saved_country_codes: bool = True
        if self.use_eu_country_codes is None:
            self.use_eu_country_codes: bool = False

        # Call to WBBaseClass without passing series-related parameters explicitly
        super().__init__(
            series_directory=self._series_directory,
            use_saved_country_codes=self.use_saved_country_codes,
            use_eu_country_codes=self.use_eu_country_codes,
            **kwargs,
        )

        # Get correct country list
        self.df_gini = self.clean_og_df()
        self.country_names_list = self.update_country_names_list_and_get_all_code_data(
            df=self.df_gini, use_saved_country_codes=self.use_saved_country_codes, use_eu_country_codes=self.use_eu_country_codes,
        )

    def reverse_coded_df_gini(self) -> pd.DataFrame:
        self.df_gini = self.add_country_codes_and_country_num_column(df=self.df_gini)
        # Reverse code Gini Coefficient so that 0 is 100 and 100 is 0
        self.df_gini[[var for var in self.df_gini.columns if 'YR' in var]] = self.df_gini[[var for var in self.df_gini.columns if 'YR' in var]].progress_apply(lambda x: (100 - x)/100)
        return self.df_gini

    def get_df_gini_and_2022_column(self) -> pd.DataFrame:
        self.df_gini = self.reverse_coded_df_gini()
        self.df_gini_2022 = self.df_gini[['country', 'country_code_3'] + [col for col in self.df_gini.columns if 'YR' in col]].copy()
        return self.df_gini, self.df_gini_2022

    def average_df_gini(self) -> pd.DataFrame:
        self.df_gini, self.df_gini_2022 = self.get_df_gini_and_2022_column()
        self.df_gini = self.df_gini[[var for var in self.df_gini.columns if 'country' in var]]\
            .merge(
                self.df_gini[[var for var in self.df_gini.columns if 'country' not in var]]\
                    .mean(axis='columns')\
                        .rename('average_gini_coefficient'),
                left_index=True,
                right_index=True,
            ).dropna().sort_values(by='country', ascending=True).reset_index(drop=True)
        self.df_gini = self.z_score_and_trichotimize(self.df_gini, 'average_gini_coefficient')
        self.df_gini = self.df_gini.merge(self.df_gini_2022, on=['country', 'country_code_3'], how='left')
        return self.df_gini

    def get_df_gini(self) -> pd.DataFrame:
        self.df_gini = self.average_df_gini()
        self.df_gini.to_csv(f'{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_directory}.csv', index=False)
        self.df_gini.to_pickle(f'{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_directory}.pkl')
        return self.df_gini

# %%
# Get Population Data
class PopulationData(WBBaseClass):
    def __init__(
        self,
        series_directory: str = None,
        use_saved_country_codes: bool = None,
        use_eu_country_codes: bool = None,
        **kwargs,
    ) -> None:
        self._series_directory: str = series_directory
        self.use_saved_country_codes: bool = use_saved_country_codes
        self.use_eu_country_codes: bool = use_eu_country_codes

        if self._series_directory is None:
            self._series_directory = 'POPULATION'
        if self.use_saved_country_codes is None:
            self.use_saved_country_codes = True
        if self.use_eu_country_codes is None:
            self.use_eu_country_codes = False
        # Call to WBBaseClass without passing series-related parameters explicitly
        super().__init__(
            series_directory=self._series_directory,
            use_saved_country_codes=self.use_saved_country_codes,
            use_eu_country_codes=self.use_eu_country_codes,
            **kwargs,
        )

        self.df = self.clean_og_df()
        self.country_names_list = self.update_country_names_list_and_get_all_code_data(
            df=self.df, use_saved_country_codes=self.use_saved_country_codes, use_eu_country_codes=self.use_eu_country_codes,
        )

    def get_df_for_year(self, year: int) -> pd.DataFrame:
        year = int(year)
        if year < self.start_year or year > self.end_year:
            raise Exception(f'Year must be between {self.start_year} and {self.end_year}.')
        self.df = self.df.loc[self.df['country'].isin(self.country_names_list)].sort_values(by='country', ascending=True).reset_index(drop=True)

        self.df_for_year = self.df[[col for col in self.df.columns if 'YR' not in col] + [f'YR{year}']].copy()
        self.df_for_year = self.z_score_and_trichotimize(self.df_for_year, f'YR{year}')
        self.df_for_year.to_csv(
            f'''{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_name.replace("'", "").strip()}_{year}.csv''', index=False,
        )
        self.df_for_year.to_pickle(
            f'''{necessary_data_path}Measures/{self.series_directory}/World Bank {self.series_name.replace("'", "").strip()}_{year}.pkl''',
        )
        return self.df_for_year

# %%
