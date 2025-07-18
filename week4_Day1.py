
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib

matplotlib.use('TkAgg') 


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.DATA_PATH = Path(r"C:\Users\游晨仪\Desktop\w4\energy.csv")
        self.OUTPUT_DIR = self.BASE_DIR / "output"
        self.create_dirs()

    def create_dirs(self):
        """创建必要的目录"""
        self.OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()


class DataLoader:
    @staticmethod
    def load_data():
        """加载能源数据集"""
        try:
            logger.info(f"正在从 {config.DATA_PATH} 加载数据...")
            df = pd.read_csv(config.DATA_PATH)

           
            if df.empty:
                raise ValueError("加载的数据为空！")

            logger.info(f"数据加载成功，形状: {df.shape}")
            logger.info(f"前5行数据:\n{df.head()}")
            logger.info(f"数据类型:\n{df.dtypes}")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

class DataAnalyzer:
    @staticmethod
    def analyze(df):
        """执行基本的数据分析"""
        logger.info("开始数据分析...")

        analysis = {
            'head': df.head(),
            'describe': df.describe(),
            'null_values': df.isnull().sum(),
            'dtypes': df.dtypes,
            'correlation': df.select_dtypes(include=['number']).corr()
        }

        logger.info("数据分析完成")
        return analysis

    @staticmethod
    def visualize(df):
        """数据可视化（确保图像显示）"""
        logger.info("生成数据可视化...")

   
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            fig1, axes = plt.subplots(nrows=len(num_cols), figsize=(10, 2*len(num_cols)))
            if len(num_cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, num_cols):
                df[col].hist(ax=ax)
                ax.set_title(f"{col} 分布")

            plt.tight_layout()
            plt.savefig(config.OUTPUT_DIR / "distributions.png")
            plt.show()  
            plt.close(fig1)
        else:
            logger.warning("没有数值列可用于分布图")

        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            fig2 = plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title("特征相关性热力图")
            plt.tight_layout()
            plt.savefig(config.OUTPUT_DIR / "correlation.png")
            plt.show() 
            plt.close(fig2)
        else:
            logger.warning("不足的数值列用于相关性热力图")


class FeatureEngineer:
    @staticmethod
    def preprocess(df, target_col=None):
        """数据预处理"""
        logger.info("开始特征工程...")

        if target_col is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[-1]
                logger.warning(f"未指定目标列，自动选择: {target_col}")
            else:
                raise ValueError("数据中没有数值列可用作目标变量")

        if target_col not in df.columns:
            raise KeyError(f"目标列 '{target_col}' 不存在。可用列: {list(df.columns)}")

        datetime_cols = df.select_dtypes(include=['object']).columns
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df = df.drop(columns=[col])
            except:
                df[col] = df[col].astype('category').cat.codes

        df = df.dropna()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        numeric_cols = X.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = (X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std()

        logger.info(f"特征工程完成 - 目标列: {target_col}")
        return X, y, target_col


class EnergyModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {'MSE': mse, 'R2 Score': r2}
        pd.DataFrame.from_dict(metrics, orient='index').to_csv(
            config.OUTPUT_DIR / "model_metrics.csv", header=['Value'])

        # 修复预测图显示
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title("预测结果 vs 真实值")
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "predictions.png")
        plt.show()  # 确保显示
        plt.close(fig)

        return metrics


def main():
    try:
        logger.info("=== 能源数据分析系统启动 ===")

        data_loader = DataLoader()
        df = data_loader.load_data()

        analyzer = DataAnalyzer()
        analysis = analyzer.analyze(df)
        analyzer.visualize(df)

        engineer = FeatureEngineer()
        X, y, target_col = engineer.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = EnergyModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        logger.info("=== 流程完成 ===")
        return analysis, metrics, target_col

    except Exception as e:
        logger.error(f"主流程出错: {e}")
        raise

if __name__ == "__main__":
    try:
        analysis_results, model_metrics, target_col = main()
        print("\n分析结果摘要:")
        print(pd.DataFrame(analysis_results['describe']))
        print(f"\n目标列: {target_col}")
        print("\n模型指标:")
        print(pd.DataFrame.from_dict(model_metrics, orient='index'))
    except Exception as e:
        print(f"程序运行出错: {e}")
