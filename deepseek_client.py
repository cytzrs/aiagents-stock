import openai
import json
from typing import Dict, List, Any, Optional
import config

class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self, model="deepseek-chat"):
        self.model = model
        self.client = openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL
        )
        
    def call_api(self, messages: List[Dict[str, str]], model: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """调用DeepSeek API"""
        # 使用实例的模型，如果没有传入则使用默认模型
        model_to_use = model or self.model
        
        # 对于 reasoner 模型，自动增加 max_tokens
        if "reasoner" in model_to_use.lower() and max_tokens <= 2000:
            max_tokens = 8000  # reasoner 模型需要更多 tokens 来输出推理过程
        
        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 处理 reasoner 模型的响应
            message = response.choices[0].message
            
            # reasoner 模型可能包含 reasoning_content（推理过程）和 content（最终答案）
            # 我们返回完整内容，包括推理过程（如果有的话）
            result = ""
            
            # 检查是否有推理内容
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                result += f"【推理过程】\n{message.reasoning_content}\n\n"
            
            # 添加最终内容
            if message.content:
                result += message.content
            
            return result if result else "API返回空响应"
            
        except Exception as e:
            return f"API调用失败: {str(e)}"
    
    def technical_analysis(self, stock_info: Dict, stock_data: Any, indicators: Dict) -> str:
        """技术面分析"""
        prompt = f"""
你是一名资深的技术分析师。请基于以下股票数据进行专业的技术面分析：

股票信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}
- 涨跌幅：{stock_info.get('change_percent', 'N/A')}%

最新技术指标：
- 收盘价：{indicators.get('price', 'N/A')}
- MA5：{indicators.get('ma5', 'N/A')}
- MA10：{indicators.get('ma10', 'N/A')}
- MA20：{indicators.get('ma20', 'N/A')}
- MA60：{indicators.get('ma60', 'N/A')}
- RSI：{indicators.get('rsi', 'N/A')}
- MACD：{indicators.get('macd', 'N/A')}
- MACD信号线：{indicators.get('macd_signal', 'N/A')}
- 布林带上轨：{indicators.get('bb_upper', 'N/A')}
- 布林带下轨：{indicators.get('bb_lower', 'N/A')}
- K值：{indicators.get('k_value', 'N/A')}
- D值：{indicators.get('d_value', 'N/A')}
- 量比：{indicators.get('volume_ratio', 'N/A')}

请从以下角度进行分析：
1. 趋势分析（均线系统、价格走势）
2. 超买超卖分析（RSI、KDJ）
3. 动量分析（MACD）
4. 支撑阻力分析（布林带）
5. 成交量分析
6. 短期、中期、长期技术判断
7. 关键技术位分析

请给出专业、详细的技术分析报告，包含风险提示。
"""
        
        messages = [
            {"role": "system", "content": "你是一名经验丰富的股票技术分析师，具有深厚的技术分析功底。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages)
    
    def fundamental_analysis(self, stock_info: Dict, financial_data: Dict = None, quarterly_data: Dict = None) -> str:
        """基本面分析"""
        
        # 构建财务数据部分
        financial_section = ""
        if financial_data and not financial_data.get('error'):
            ratios = financial_data.get('financial_ratios', {})
            if ratios:
                financial_section = f"""
详细财务指标：
【盈利能力】
- 净资产收益率(ROE)：{ratios.get('净资产收益率ROE', ratios.get('ROE', 'N/A'))}
- 总资产收益率(ROA)：{ratios.get('总资产收益率ROA', ratios.get('ROA', 'N/A'))}
- 销售毛利率：{ratios.get('销售毛利率', ratios.get('毛利率', 'N/A'))}
- 销售净利率：{ratios.get('销售净利率', ratios.get('净利率', 'N/A'))}

【偿债能力】
- 资产负债率：{ratios.get('资产负债率', 'N/A')}
- 流动比率：{ratios.get('流动比率', 'N/A')}
- 速动比率：{ratios.get('速动比率', 'N/A')}

【运营能力】
- 存货周转率：{ratios.get('存货周转率', 'N/A')}
- 应收账款周转率：{ratios.get('应收账款周转率', 'N/A')}
- 总资产周转率：{ratios.get('总资产周转率', 'N/A')}

【成长能力】
- 营业收入同比增长：{ratios.get('营业收入同比增长', ratios.get('收入增长', 'N/A'))}
- 净利润同比增长：{ratios.get('净利润同比增长', ratios.get('盈利增长', 'N/A'))}

【每股指标】
- 每股收益(EPS)：{ratios.get('EPS', 'N/A')}
- 每股账面价值：{ratios.get('每股账面价值', 'N/A')}
- 股息率：{ratios.get('股息率', stock_info.get('dividend_yield', 'N/A'))}
- 派息率：{ratios.get('派息率', 'N/A')}
"""
            
            # 添加报告期信息
            if ratios.get('报告期'):
                financial_section = f"\n财务数据报告期：{ratios.get('报告期')}\n" + financial_section
        
        # 构建季报数据部分
        quarterly_section = ""
        if quarterly_data and quarterly_data.get('data_success'):
            # 使用格式化的季报数据
            from quarterly_report_data import QuarterlyReportDataFetcher
            fetcher = QuarterlyReportDataFetcher()
            quarterly_section = f"""

【最近8期季报详细数据】
{fetcher.format_quarterly_reports_for_ai(quarterly_data)}

以上是通过akshare获取的最近8期季度财务报告，请重点基于这些数据进行趋势分析。
"""
        
        prompt = f"""
你是一名资深的基本面分析师，拥有CFA资格和10年以上的证券分析经验。请基于以下详细信息进行深入的基本面分析：

【基本信息】
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}
- 市值：{stock_info.get('market_cap', 'N/A')}
- 行业：{stock_info.get('sector', 'N/A')}
- 细分行业：{stock_info.get('industry', 'N/A')}

【估值指标】
- 市盈率(PE)：{stock_info.get('pe_ratio', 'N/A')}
- 市净率(PB)：{stock_info.get('pb_ratio', 'N/A')}
- 市销率(PS)：{stock_info.get('ps_ratio', 'N/A')}
- Beta系数：{stock_info.get('beta', 'N/A')}
- 52周最高：{stock_info.get('52_week_high', 'N/A')}
- 52周最低：{stock_info.get('52_week_low', 'N/A')}
{financial_section}
{quarterly_section}

请从以下维度进行专业、深入的分析：

1. **公司质地分析**
   - 业务模式和核心竞争力
   - 行业地位和市场份额
   - 护城河分析（品牌、技术、规模等）

2. **盈利能力分析**
   - ROE和ROA水平评估
   - 毛利率和净利率趋势
   - 与行业平均水平对比
   - 盈利质量和持续性

3. **财务健康度分析**
   - 资产负债结构
   - 偿债能力评估
   - 现金流状况
   - 财务风险识别

4. **成长性分析**
   - 收入和利润增长趋势
   - 增长驱动因素
   - 未来成长空间
   - 行业发展前景

5. **季报趋势分析（如有季报数据）** ⭐ 重点分析
   - **营收趋势**：分析最近8期营业收入的变化趋势，识别增长或下滑
   - **利润趋势**：分析净利润和每股收益的变化，评估盈利能力变化
   - **现金流分析**：经营现金流、投资现金流、筹资现金流的变化趋势
   - **资产负债变化**：资产规模、负债水平、所有者权益的变化
   - **季度环比/同比**：计算关键指标的环比和同比变化率
   - **经营质量**：评估收入质量、利润质量、现金流质量
   - **异常识别**：识别异常波动，分析原因（季节性、一次性事件等）
   - **趋势预判**：基于最近8期数据预判未来1-2个季度趋势

6. **估值分析**
   - 当前估值水平（PE、PB）
   - 历史估值区间对比
   - 行业估值对比
   - 结合季报趋势调整估值预期
   - 合理估值区间判断

7. **投资价值判断**
   - 综合评分（0-100分）
   - 投资亮点（特别关注季报改善趋势）
   - 投资风险（关注季报恶化信号）
   - 适合的投资者类型

**分析要求：**
- 如果有季报数据，请重点分析8期数据的趋势变化
- 识别改善或恶化的早期信号
- 结合季报数据对未来业绩进行预判
- 数据分析要深入，结论要有依据
- 结合当前市场环境和行业发展趋势

请给出专业、详细的基本面分析报告。
"""
        
        messages = [
            {"role": "system", "content": "你是一名经验丰富的股票基本面分析师，擅长公司财务分析和行业研究。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages)
    
    def fund_flow_analysis(self, stock_info: Dict, indicators: Dict, fund_flow_data: Dict = None) -> str:
        """资金面分析"""
        
        # 构建资金流向数据部分 - 使用akshare格式化数据
        fund_flow_section = ""
        if fund_flow_data and fund_flow_data.get('data_success'):
            # 使用格式化的资金流向数据
            from fund_flow_akshare import FundFlowAkshareDataFetcher
            fetcher = FundFlowAkshareDataFetcher()
            fund_flow_section = f"""

【近20个交易日资金流向详细数据】
{fetcher.format_fund_flow_for_ai(fund_flow_data)}

以上是通过akshare从东方财富获取的实际资金流向数据，请重点基于这些数据进行趋势分析。
"""
        else:
            fund_flow_section = "\n【资金流向数据】\n注意：未能获取到资金流向数据，将基于成交量进行分析。\n"
        
        prompt = f"""
你是一名资深的资金面分析师，擅长从资金流向数据中洞察主力行为和市场趋势。

【基本信息】
股票代码：{stock_info.get('symbol', 'N/A')}
股票名称：{stock_info.get('name', 'N/A')}
当前价格：{stock_info.get('current_price', 'N/A')}
市值：{stock_info.get('market_cap', 'N/A')}

【技术指标】
- 量比：{indicators.get('volume_ratio', 'N/A')}
- 当前成交量与5日均量比：{indicators.get('volume_ratio', 'N/A')}
{fund_flow_section}

【分析要求】

请你**基于上述近20个交易日的完整资金流向数据**，从以下角度进行深入分析：

1. **资金流向趋势分析** ⭐ 重点
   - 分析近20个交易日主力资金的累计净流入/净流出
   - 识别资金流向的趋势性特征（持续流入、持续流出、震荡）
   - 计算主力资金净流入天数占比
   - 评估资金流向强度（累计金额、平均每日金额）

2. **主力资金行为分析** ⭐ 核心重点
   - **主力资金总体表现**：累计净流入金额、占比、趋势方向
   - **超大单分析**：机构大资金的进出动作
   - **大单分析**：主力资金的操作特征
   - **主力操作意图研判**：
     * 吸筹建仓：持续净流入 + 股价上涨/盘整
     * 派发出货：持续净流出 + 股价下跌/高位
     * 洗盘整理：震荡流入流出 + 股价调整
     * 拉升推动：集中大额流入 + 股价快速上涨

3. **散户资金行为分析**
   - **中单、小单的动向**：散户的买卖情绪
   - **主力与散户博弈**：
     * 主力流入、散户流出 → 专业资金吸筹
     * 主力流出、散户流入 → 高位接盘风险
     * 同向流动 → 趋势明确
   - 散户参与度和情绪判断

4. **量价配合分析**
   - 资金流向与股价涨跌的配合度
   - 识别量价背离：
     * 价涨量缩 + 资金流出 → 警惕顶部
     * 价跌量增 + 资金流入 → 可能见底
   - 成交活跃度变化趋势

5. **关键信号识别**
   - **买入信号**：
     * 主力持续净流入
     * 大单明显流入
     * 资金流入 + 股价上涨
   - **卖出信号**：
     * 主力持续净流出
     * 大额资金出逃
     * 资金流出 + 股价滞涨或下跌
   - **观望信号**：
     * 资金流向不明确
     * 主力与散户博弈激烈

6. **阶段性特征**
   - 早期阶段（前10个交易日）vs 近期阶段（后10个交易日）
   - 资金流向的变化趋势
   - 转折点识别

7. **投资建议**
   - 基于资金流向的操作建议
   - 关注重点和风险提示
   - 资金面对后市的指示意义
   - 未来资金流向预判

8. **投资建议**
   - 基于资金面的明确操作建议
   - 买入/持有/卖出的判断依据
   - 仓位管理建议

【分析原则】
- 主力资金持续流入 + 股价上涨 → 强势信号，主力看好
- 主力资金流出 + 股价上涨 → 警惕信号，可能是散户接盘
- 主力资金流入 + 股价下跌 → 可能是主力低位吸筹
- 主力资金流出 + 股价下跌 → 弱势信号，主力看空
- 注意区分短期波动与趋势性变化

请给出专业、详细、有深度的资金面分析报告。记住：要基于问财数据的实际内容进行分析，而不是假设！
"""
        
        messages = [
            {"role": "system", "content": "你是一名经验丰富的资金面分析师，擅长市场资金流向和主力行为分析，能够深入解读资金数据背后的投资逻辑。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages, max_tokens=3000)
    
    def comprehensive_discussion(self, technical_report: str, fundamental_report: str, 
                               fund_flow_report: str, stock_info: Dict) -> str:
        """综合讨论"""
        prompt = f"""
现在需要进行一场投资决策会议，你作为首席分析师，需要综合各位分析师的报告进行讨论。

股票基本信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}

技术面分析报告：
{technical_report}

基本面分析报告：
{fundamental_report}

资金面分析报告：
{fund_flow_report}

请作为首席分析师，综合以上三个维度的分析报告，进行深入讨论：

1. 各个分析维度的一致性和分歧点
2. 不同分析结论的权重考量
3. 当前市场环境下的投资逻辑
4. 潜在风险和机会识别
5. 不同投资周期的考量（短期、中期、长期）
6. 市场情绪和预期管理

请模拟一场专业的投资讨论会议，体现不同观点的碰撞和融合。
"""
        
        messages = [
            {"role": "system", "content": "你是一名资深的首席投资分析师，擅长综合不同维度的分析形成投资判断。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages, max_tokens=6000)
    
    def chan_analysis(self, stock_info: Dict, stock_data: Any, indicators: Dict) -> str:
        """缠论分析"""
        prompt = f"""
你是一名深谙“缠中说禅”理论精髓的实战派分析师。请以“走势终完美”为根本指导原则，结合走势形态（中枢、笔、段）
与动力学（背驰、买卖点），对指定股票进行多级别联立分析，并给出明确的当下策略：

股票信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}
- 涨跌幅：{stock_info.get('change_percent', 'N/A')}%

最新技术指标：
- 收盘价：{indicators.get('price', 'N/A')}
- MA5：{indicators.get('ma5', 'N/A')}
- MA10：{indicators.get('ma10', 'N/A')}
- MA20：{indicators.get('ma20', 'N/A')}
- MA60：{indicators.get('ma60', 'N/A')}
- RSI：{indicators.get('rsi', 'N/A')}
- MACD：{indicators.get('macd', 'N/A')}
- MACD信号线：{indicators.get('macd_signal', 'N/A')}
- 布林带上轨：{indicators.get('bb_upper', 'N/A')}
- 布林带下轨：{indicators.get('bb_lower', 'N/A')}
- K值：{indicators.get('k_value', 'N/A')}
- D值：{indicators.get('d_value', 'N/A')}
- 量比：{indicators.get('volume_ratio', 'N/A')}

请从以下角度进行分析：
1. 界定“走势形态”与“走势终完美”的关系
   在分析前，请简明阐述你的分析哲学：
   1.2,“走势形态”（中枢、笔、段）是几何学，用以描述走势的结构。
   1.3,“走势终完美”是生态学，它保证了任何走势形态都必然有始有终。
   1.3,我们的工作是：通过观察“走势形态”的构建，来推断“走势终完美”这一过程在何时、何级别上即将或已经实现。
2. 多级别走势结构分析（从大到小）
   请严格按照以下级别进行分析，高级别是战略方向，低级别是战术切入点。
   2.1 日线级别（战略视角）：
    2.1.1 当前走势类型：判断当前是处于上涨、下跌还是盘整中？
    2.1.2 中枢定位：当前走势在构筑新的中枢，还是围绕既有的中枢震荡？请画出核心中枢区间。
    2.1.3 “完美”状态评估：根据“走势终完美”，当前趋势是否显示出衰竭迹象（例如，进入背驰段）？一个趋势的“完美”（完成），通常以一个背驰信号为确认。
   2.2 30分钟级别（战役视角）：
    2.2.1 与日线关系：当前30分钟走势，是日线中的哪一笔？
    2.2.2 中枢与背驰：在这30分钟一笔的内部，是否形成了中枢？是否发生了背驰？这里是发现日线一笔是否“完美”的关键区域。
   2.3 5分钟级别（战术视角）：
    2.3.1 精确定位：用于验证30分钟一笔的结束点，并寻找最具安全边际的买卖点。
    2.3.2 买卖点萌芽：第三类买卖点通常在此级别率先形成，为高级别走势的确认提供早期信号。
3. 综合研判与“走势完美”的当下判断
   这是分析的核心结论部分。
   3.1 “走势终完美”的当下结论：
    3.1.1 “已然完美”：如果高级别（如日线）的上涨/下跌趋势已经确认背驰，并且低级别（如30分钟）已经走出了反向的走势类型，则可以判断原趋势“已然完美”。操作上，应寻找反弹/回调结束后的反向开仓机会。
    3.1.2 “即将完美”：如果高级别进入背驰段，且低级别（如5分钟）开始出现动能衰竭（如小级别背驰），则原趋势“即将完美”。操作上，应准备在高点/低点区域了结头寸，并密切关注反转信号。
    3.1.3 “完美之后”：原趋势“完美”完成后，新的走势类型是什么？是盘整还是反向趋势？这决定了后续的操作节奏和空间。
   3.2 三类买卖点定位：
    3.2.1 基于以上判断，明确指出当前是否存在或正在酝酿哪一类买卖点。
    3.2.2 例如：“日线上涨趋势已确认‘完美’，当前30分钟下跌走势也‘完美’（形成中枢且背驰），由此产生的30分钟第一类买点，即是日线上的第二类买点。”
4. 动态推演与策略计划
    4.1 完全分类与应对：对后续走势进行“完全分类”，并制定应对策略。
        4.1.1 最优情况：走势按预期发展，买点后强势上涨，形成第三类买点则持有或加仓。
        4.1.2 次优情况：买点后进入中枢盘整，耐心持有，等待突破。
        4.1.3 最坏情况：走势破坏预期（如跌破关键止损位），说明判断错误，“走势终完美”以另一种方式实现（例如，以更深的下跌来完成），必须严格执行止损。
    4.2 关键位置与操作：
        4.2.1 买卖点：明确的位置与级别。
        4.2.2 止损位：设置在能证明原买卖点无效的位置。
        4.2.3 仓位建议：根据买卖点的级别（日线级重，5分钟级轻）和信号强弱给出。

    请给出专业、详细的技术分析报告，包含风险提示。

"""
        
        messages = [
            {"role": "system", "content": "你是一名深谙“缠中说禅”理论精髓的实战派分析师，具有深厚的缠论分析功底。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages)
    
    def cgyj_analysis(self, stock_info: Dict, stock_data: Any, indicators: Dict) -> str:
        """养家心法分析"""
        prompt = f"""
你是一位深谙“炒股养家”心法的资深游资操盘手。你的核心哲学是：“心中有预期，操作看信号，永远站在情绪和资金的一边。” 
请严格遵循心法要义，对指定股票进行短线交易分析，并制定一份可直接执行的交易计划。：

股票信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}
- 涨跌幅：{stock_info.get('change_percent', 'N/A')}%

最新技术指标：
- 收盘价：{indicators.get('price', 'N/A')}
- MA5：{indicators.get('ma5', 'N/A')}
- MA10：{indicators.get('ma10', 'N/A')}
- MA20：{indicators.get('ma20', 'N/A')}
- MA60：{indicators.get('ma60', 'N/A')}
- RSI：{indicators.get('rsi', 'N/A')}
- MACD：{indicators.get('macd', 'N/A')}
- MACD信号线：{indicators.get('macd_signal', 'N/A')}
- 布林带上轨：{indicators.get('bb_upper', 'N/A')}
- 布林带下轨：{indicators.get('bb_lower', 'N/A')}
- K值：{indicators.get('k_value', 'N/A')}
- D值：{indicators.get('d_value', 'N/A')}
- 量比：{indicators.get('volume_ratio', 'N/A')}

请从以下角度进行分析：
1. 势——审视市场环境与题材逻辑
    1. 市场情绪周期定位
    判断当前市场处于 "启动—发酵—高潮—退潮—冰点" 中的哪个阶段？
    市场的赚钱效应和亏钱效应分别集中在哪些板块？
    当前市场的最高连板数和涨停家数反映了怎样的风险偏好？
    2. 个股题材强度分析
    该股的核心题材是什么？（例如：新技术、新政策、重大事件）
    该题材的 "新、大、热" 程度如何？是主线题材，还是支线/过渡性题材？
    评估题材的预期寿命和想象空间。
2. 道——剖析资金动向与筹码结构
    1. 资金攻击意图
    近期是否有 "点火"（直线拉升）、"封板"、"炸板回封" 等关键资金行为？
    分析量价关系：是缩量加速？放量分歧？还是无量阴跌？
    2. 筹码博弈状态
    当前股价处于何种位置？（低位启动、中位换手、高位震荡、下跌通道）
    关键压力位（前高、密集套牢区）和支撑位（前低、启动平台）在哪里？
3. 术——制定买卖策略与风控计划
    1. 买入策略（"心中的预期"）
    "确定性"买点：在何种信号出现时，代表确定性最高？ 示例： "带量高开并快速拉红，确认弱转强"、"爆量分歧后，首次回封涨停时"
    "风报比"买点：在何种情况下，可以小仓位博弈高风报比？ 示例： "回调至关键支撑位且分时图出现资金承接"
    2. 卖出策略（"操作看信号"）
    主动止盈点：达到何种条件即可锁定利润？ 示例： "封单急剧减少且板块出现大面股时，分批卖出"
    无条件止损点：出现何种信号必须果断离场？ 示例： "买入后次日低开低走，跌破昨日最低点"
    3. 仓位管理
    根据本次交易的 "确定性等级"（高、中、低），给出具体的仓位建议 示例： "3成仓试错、5成仓主攻、1成仓娱乐"
4. 律——明确交易纪律与推演
    1. 交易纪律重申
    必须无条件执行的1-2条核心纪律 示例： "宁可错过，不可做错"、"止损单必须开盘前挂好"
    2. 完全分类与应对
    对次日走势进行 "强、中、弱" 三种完全分类：

    走势分类	应对策略
    超预期（强）	如何持有或加仓？
    符合预期（中）	如何观察和持有？
    低于预期（弱）	如何果断卖出或减仓？

请给出专业、详细的技术分析报告，包含风险提示。
"""
        
        messages = [
            {"role": "system", "content": "基于养家心法哲学，生成包含明确买卖点和风控措施的短期交易计划。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages)
    
    def final_decision(self, comprehensive_discussion: str, stock_info: Dict, 
                      indicators: Dict) -> Dict[str, Any]:
        """最终投资决策"""
        prompt = f"""
基于前期的综合分析讨论，现在需要做出最终的投资决策。

股票信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}

综合分析讨论结果：
{comprehensive_discussion}

当前关键技术位：
- MA20：{indicators.get('ma20', 'N/A')}
- 布林带上轨：{indicators.get('bb_upper', 'N/A')}
- 布林带下轨：{indicators.get('bb_lower', 'N/A')}

请给出最终投资决策，必须包含以下内容：

1. 投资评级：买入/持有/卖出
2. 目标价位（具体数字）
3. 操作建议（具体的买入/卖出策略）
4. 进场位置（具体价位区间）
5. 止盈位置（具体价位）
6. 止损位置（具体价位）
7. 持有周期建议
8. 风险提示
9. 仓位建议（轻仓/中等仓位/重仓）

请以JSON格式输出决策结果，格式如下：
{{
    "rating": "买入/持有/卖出",
    "target_price": "目标价位数字",
    "operation_advice": "具体操作建议",
    "entry_range": "进场价位区间",
    "take_profit": "止盈价位",
    "stop_loss": "止损价位",
    "holding_period": "持有周期",
    "position_size": "仓位建议",
    "risk_warning": "风险提示",
    "confidence_level": "信心度(1-10分)"
}}
"""
        
        messages = [
            {"role": "system", "content": "你是一名专业的投资决策专家，需要给出明确、可执行的投资建议。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.call_api(messages, temperature=0.3, max_tokens=4000)
        
        try:
            # 尝试解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision_json = json.loads(json_match.group())
                return decision_json
            else:
                # 如果无法解析JSON，返回文本响应
                return {"decision_text": response}
        except:
            return {"decision_text": response}

