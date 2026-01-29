import json
import os
import logging
import difflib
import numpy as np
import pandas as pd
import random
import requests
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from scipy.stats import skellam, poisson
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

TEAM_MAPPING = {
    "man utd": "manchester united",
    "man city": "manchester city",
    "tottenham hotspur": "tottenham",
    "nottm forest": "nottingham forest",
    "sheff utd": "sheffield united",
    "west ham": "west ham united",
    # Добавь сюда остальные команды, если они были
}

# --- НОВЫЙ МЕТОД: ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ РАСЧЕТА МАРЖИ ---
def calculate_margin_global(odds_list: list) -> float:
    """Считает маржу букмекера на основе списка коэффициентов"""
    try:
        valid_odds = [float(o) for o in odds_list if o and float(o) > 1]
        if not valid_odds:
            return 0.0
        return sum(1/o for o in valid_odds) - 1.0
    except:
        return 0.0

@dataclass
class TeamRating:
    att_raw: float
    def_raw: float
    att_norm: float
    def_norm: float
    real_pts_avg: float
    matches_played: int
    quality: float
    stability: float
    hfa: float = 1.0


class TitaniumBrain:
    def __init__(self):
        self.roi = 0.0
        self.winstreak = 0
        self.phrases = {
            "greeting_good": [
                "✅ СИСТЕМА АКТИВНА. Текущая доходность портфеля положительная.",
                "📈 STATUS: GREEN. Модель откалибрована. Фиксируется положительный ROI.",
                "💎 ANALYTICS CORE ONLINE. Эффективность стратегии в пределах нормы.",
            ],
            "greeting_bad": [
                "⚠️ ВНИМАНИЕ: Зафиксирована просадка депозита. Рекомендуется снижение рисков.",
                "📉 STATUS: AMBER. Волатильность рынка повышена. Требуется строгая селекция.",
                "🛡️ SYSTEM ALERT: Отрицательная динамика PnL. Активирован защитный протокол.",
            ],
            "value_huge": [
                "⚡ ОБНАРУЖЕНА РЫНОЧНАЯ НЕЭФФЕКТИВНОСТЬ. Значительное отклонение от линии БК.",
                "📊 STRONG SIGNAL. Высокая вероятность недооценки исхода букмекером.",
                "💎 ALPHA DETECTED. Математическое ожидание существенно выше нормы.",
            ],
            "value_ok": [
                "✅ MODERATE VALUE. Выявлен умеренный перевес вероятности.",
                "📈 RATING: BUY. Коэффициент превышает расчетную справедливую цену.",
            ],
            "no_value": [
                "🚫 NO SIGNAL. Котировки рынка соответствуют расчетным вероятностям.",
                "📉 EFFICIENT MARKET. Перевес отсутствует. Инвестиция не рекомендована.",
            ],
        }

    def set_stats(self, roi, winstreak):
        self.roi = roi
        self.winstreak = winstreak

    def say_hello(self):
        key = "greeting_good" if self.roi >= 0 else "greeting_bad"
        return random.choice(self.phrases[key]) + f" [YIELD: {self.roi:+.2f}%]"


class MatchPredictor:
    def __init__(self, data_dir: str):
        self.cfg = {
            "api_key": "8a25e55c920042d09149a9b37b6a2356",
            "default_xg": 1.35,
            "rho": -0.11,
            "max_injury_frac": 0.45,
            "xg_weights": {"S": 0.02, "ST": 0.12, "P": 0.0, "C": 0.01},
            "kelly_fraction": 0.1,
            "max_bank_pct": 0.05,
            "decay_rate": 0.05,
            "fatigue_penalty": 0.12,
        }
        self.last_match_dates: Dict[str, datetime] = {}
        self.brain = TitaniumBrain()
        self.bankroll = 1000.0
        self.ratings: Dict[str, TeamRating] = {}
        self.injuries: Dict[str, Dict[str, float]] = {}
        self.league_mean_xg = self.cfg["default_xg"]
        self.home_advantage = 1.15

        self.fpl_id_map = {
            1: "arsenal",
            2: "aston villa",
            3: "bournemouth",
            4: "brentford",
            5: "brighton",
            6: "chelsea",
            7: "crystal palace",
            8: "everton",
            9: "fulham",
            10: "ipswich",
            11: "leicester",
            12: "liverpool",
            13: "man city",
            14: "man utd",
            15: "newcastle",
            16: "nottm forest",
            17: "southampton",
            18: "tottenham",
            19: "west ham",
            20: "wolves",
        }

        # Карта турниров для мониторинга усталости
        self.tournaments_config = {
            "PL": {"id": 2021, "file": "pl.json", "label": "АПЛ"},
            "CL": {"id": 2001, "file": "cl.json", "label": "Лига Чемпионов"},
            "BL1": {"id": 2002, "file": "bl1.json", "label": "Бундеслига"},
            "PD": {"id": 2014, "file": "pd.json", "label": "Ла Лига"},
            "SA": {"id": 2019, "file": "sa.json", "label": "Серия А"},
            "FL1": {"id": 2015, "file": "fl1.json", "label": "Лига 1"}
        }
        # Путь к папке, где будут лежать файлы всех кубков
        self.cups_dir = os.path.join(data_dir, "history_data")
        self.data_dir = data_dir
        self.odds_dir = os.path.join(data_dir, "history_odds")
        self.fixtures_dir = os.path.join(self.data_dir, "history_data", "fixtures")
        os.makedirs(self.fixtures_dir, exist_ok=True)
        self.injuries_path = os.path.join(
            data_dir, "history_Players", "fpl_players_injuries.json"
        )
        self.upcoming_path = os.path.join(
            data_dir, "history_data", "upcoming_fixtures.json"
        )
        self.history_path = os.path.join(data_dir, "prediction_history.json")

        self.initialize()

    def download_all_football_data(self):
        """[TITANIUM SYNC] Авто-загрузка результатов всех 6 турниров через API"""
        logging.info("📡 Синхронизация Titanium Core (API: football-data.org)...")
        
        # 1. Обновляем FPL (травмы) - работает без ключа
        fpl_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        try:
            r = requests.get(fpl_url, timeout=10)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(self.injuries_path), exist_ok=True)
                with open(self.injuries_path, "w", encoding="utf-8") as f:
                    json.dump(r.json(), f, indent=4, ensure_ascii=False)
                logging.info("✅ Травмы FPL обновлены.")
        except Exception as e:
            logging.error(f"❌ Ошибка FPL: {e}")

        # 2. Загрузка результатов лиг (требует ключ)
        token = self.cfg.get("api_key")
        if not token or "ТВОЙ_КЛЮЧ" in token:
            logging.warning("⚠️ API Key не настроен. Пропускаем загрузку результатов лиг.")
            return False

        headers = {'X-Auth-Token': token}
        for name, info in self.tournaments_config.items():
            url = f"https://api.football-data.org/v4/competitions/{info['id']}/matches?status=FINISHED"
            try:
                time.sleep(1.5) # Пауза для бесплатного тарифа
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json().get("matches", [])
                    target_path = os.path.join(self.fixtures_dir, info['file'])
                    with open(target_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    logging.info(f"✅ {name}: Загружено матчей.")
                elif response.status_code == 429:
                    logging.warning("⏳ Лимит API запросов превышен.")
                    break 
            except Exception as e:
                logging.error(f"❌ Ошибка {name}: {e}")
        return True

    def canonicalize(self, name, silent=False):
        if not name:
            return ""
        name = name.strip().lower()
        if name in TEAM_MAPPING:
            return TEAM_MAPPING[name]
        if self.ratings:
            matches = difflib.get_close_matches(
                name, list(self.ratings.keys()), n=1, cutoff=0.7
            )
            if matches:
                return matches[0]
        return name

    def _get_fatigue_factor(self, team_name: str) -> float:
        """[TITANIUM METHOD] Анализ свежести команды по всем 6 турнирам"""
        team_c = self.canonicalize(team_name)
        now = datetime.now()
        last_game_date = None

        # Проходим по всем файлам, указанным в tournaments_map
        for t_name, info in self.tournaments_config.items():
            path = os.path.join(self.fixtures_dir, info['file'])
            if not os.path.exists(path):
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    matches = json.load(f)
                    
                    for m in matches: 
                        # ... тут твой код извлечения имен команд ...
                        h_raw = m.get("homeTeam", {}).get("name") or m.get("home") or m.get("team1")
                        a_raw = m.get("awayTeam", {}).get("name") or m.get("away") or m.get("team2")
                        
                        if not h_raw or not a_raw: continue
                        
                        h = self.canonicalize(str(h_raw))
                        a = self.canonicalize(str(a_raw))
                        
                        if (team_c == h or team_c == a) and str(m.get("status")).upper() in ["FINISHED", "FT", "ЗАВЕРШЕН"]:
                            raw_date = m.get("date") or m.get("utcDate") or m.get("start_time")
                            if not raw_date: continue
                            
                            try:
                                # Преобразуем дату
                                m_date = pd.to_datetime(raw_date).tz_localize(None)
                                if last_game_date is None or m_date > last_game_date:
                                    last_game_date = m_date
                            except: # Вот этот except ОБЯЗАТЕЛЕН для каждого try
                                continue
            except Exception as e: # И этот тоже для внешнего блока чтения файла
                logging.error(f"Ошибка при чтении файла {path}: {e}")
                continue

        if not last_game_date:
            return 1.0 # Нет данных о матчах — считаем команду свежей

        days_rest = (now - last_game_date).total_seconds() / 86400

        if days_rest < 0: days_rest = 7
        
        # Штрафы Titanium версии
        if days_rest < 2: # Если играли буквально вчера-позавчера
            logging.info(f"⚠️ {team_name} экстремальная усталость ({days_rest:.1f} дн. отдыха)")
            return 0.94 # Смягчаем штраф, так как у топ-клубов есть ротация
        if days_rest < 4: 
            return 0.97
            
        return 1.0

    def safe_float(self, prompt: str, can_skip: bool = False) -> Optional[float]:
        while True:
            raw = input(prompt).replace(",", ".").strip()
            if not raw:
                if can_skip:
                    return None
                continue
            try:
                return float(raw)
            except ValueError:
                print("⚠️ Введите число или оставьте пустым для пропуска")

    def safe_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
        return pd.Series([0.0] * len(df), index=df.index)
    
    def auto_update_db(self):
        """Скачивает данные за последние 10 лет для глубокого анализа."""
        logging.info("📚 Анализ исторической базы (глубина: 10 лет)...")
        
        now = datetime.now()
        # Определяем год начала текущего футбольного сезона
        current_start_year = now.year - 1 if now.month < 7 else now.year
        
        downloaded_count = 0
        # Цикл от 0 до 9 (проход по 10 сезонам назад)
        for i in range(10):
            start = current_start_year - i
            end = start + 1
            # Формируем код сезона: 2025/2026 превращается в '2526'
            season_code = f"{str(start)[-2:]}{str(end)[-2:]}"
            
            url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
            filename = f"E0_{season_code}.csv"
            file_path = os.path.join(self.odds_dir, filename)

            # Если файла за этот год еще нет — скачиваем
            if not os.path.exists(file_path):
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        logging.info(f"📥 Загружен сезон {start}/{end} (Код: {season_code})")
                        downloaded_count += 1
                    else:
                        # Некоторые очень старые сезоны могут иметь другой формат URL, 
                        # но для последних 10 лет этот стандарт работает.
                        logging.warning(f"⚠️ Сезон {season_code} не найден на сервере.")
                except Exception as e:
                    logging.error(f"❌ Ошибка загрузки сезона {season_code}: {e}")
        
        if downloaded_count == 0:
            logging.info("✅ Все исторические данные (10 лет) актуальны.")
        else:
            logging.info(f"🚀 База обновлена. Добавлено файлов: {downloaded_count}")

    def initialize(self):
        logging.info("🚀 System initialize...")
        for path in [
            self.odds_dir,
            os.path.dirname(self.injuries_path),
            os.path.dirname(self.upcoming_path),
        ]:
            os.makedirs(path, exist_ok=True)
        self.auto_update_db()
        self.download_all_football_data()
        self.ratings.clear()
        self.injuries.clear()
        self.load_historical_stats()
        self.settle_bets()
        self.load_injuries_real()
        self.sync_bankroll()

    def sync_bankroll(self):
        base_bank = 1000.0
        total_pnl = 0.0
        wins = 0
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    total_pnl = sum(item.get("pnl", 0.0) for item in history)
                    for item in reversed(history):
                        if item.get("pnl", 0) > 0:
                            wins += 1
                        else:
                            break
            except Exception:
                pass
        self.bankroll = base_bank + total_pnl
        roi = (total_pnl / base_bank) * 100
        self.brain.set_stats(roi, wins)
        logging.info(f"📊 Bankroll synced: {self.bankroll:.2f} | ROI: {roi:+.2f}%")

    def load_historical_stats(self):
        now = datetime.now()
        if not os.path.exists(self.odds_dir):
            logging.warning("⚠️ Папка с кэфами не найдена!")
            return

        files = sorted(
            [f for f in os.listdir(self.odds_dir) if f.startswith("E0") or f.startswith("E1")],
            reverse=True,
        )
        all_dfs = []
        w_cfg = self.cfg["xg_weights"]

        for f in files:
            try:
                path = os.path.join(self.odds_dir, f)
                df = pd.read_csv(path).dropna(subset=["FTHG", "FTAG"])
                league_coeff = 0.85 if f.startswith("E1") else 1.0

                # Исправленный блок обработки дат
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format='mixed')
                
                # ВОТ ТУТ БЫЛА ОШИБКА: ОПРЕДЕЛЯЕМ days_old
                diffs = (now - df["Date"]).dt.days
                days_old = diffs.fillna(100) 
                
                # Теперь считаем вес (W = e^(-0.0005 * t))
                df["weight"] = np.exp(-0.0005 * days_old)

                # Расчет xG
                df["h_xg"] = (w_cfg["S"] * self.safe_col(df, "HS") + 
                              w_cfg["ST"] * self.safe_col(df, "HST") + 
                              w_cfg["C"] * self.safe_col(df, "HC")) * league_coeff
                df["a_xg"] = (w_cfg["S"] * self.safe_col(df, "AS") + 
                              w_cfg["ST"] * self.safe_col(df, "AST") + 
                              w_cfg["C"] * self.safe_col(df, "AC")) * league_coeff

                all_dfs.append(df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "h_xg", "a_xg", "weight"]])
            except Exception as e:
                logging.warning(f"❌ Ошибка файла {f}: {e}")

        if not all_dfs:
            logging.error("⛔ Ни один файл не был загружен! Проверь формат CSV.")
            return

        played = pd.concat(all_dfs).drop_duplicates()

        # Среднее по лиге
        total_w = played["weight"].sum()
        mean_h = (played["h_xg"] * played["weight"]).sum() / total_w
        mean_a = (played["a_xg"] * played["weight"]).sum() / total_w
        historical_mean = (mean_h + mean_a) / 2
        self.league_mean_xg = max(1.35, historical_mean)
        self.home_advantage = np.clip(mean_h / mean_a, 1.0, 1.35)

        stats = {}
        for _, row in played.iterrows():
            h_c = self.canonicalize(row["HomeTeam"], True)
            a_c = self.canonicalize(row["AwayTeam"], True)
            wt = row["weight"]

            for team in [h_c, a_c]:
                if team not in stats:
                    stats[team] = {'xf': 0, 'xa': 0, 'gs': 0, 'pts': 0, 'w_sum': 0, 'count': 0}

            h_p = 3 if row["FTHG"] > row["FTAG"] else (1 if row["FTHG"] == row["FTAG"] else 0)
            a_p = 3 if row["FTAG"] > row["FTHG"] else (1 if row["FTAG"] == row["FTHG"] else 0)

            stats[h_c]['xf'] += row['h_xg'] * wt
            stats[h_c]['xa'] += row['a_xg'] * wt
            stats[h_c]['gs'] += row['FTHG'] * wt
            stats[h_c]['pts'] += h_p * wt
            stats[h_c]['w_sum'] += wt
            stats[h_c]['count'] += 1

            stats[a_c]['xf'] += row['a_xg'] * wt
            stats[a_c]['xa'] += row['h_xg'] * wt
            stats[a_c]['gs'] += row['FTAG'] * wt
            stats[a_c]['pts'] += a_p * wt
            stats[a_c]['w_sum'] += wt
            stats[a_c]['count'] += 1

        for team, s in stats.items():
            ws = s['w_sum'] + 1e-9
            avg_xf = s['xf'] / ws
            avg_xa = s['xa'] / ws
            quality = np.clip((s['gs'] / ws) / (avg_xf + 1e-9), 0.85, 1.20)
            stability = np.clip(np.sqrt(s['count']) / np.sqrt(15), 0.75, 1.0)

            self.ratings[team] = TeamRating(
                att_raw=float(avg_xf),
                def_raw=float(avg_xa),
                att_norm=float(avg_xf / self.league_mean_xg),
                def_norm=float(avg_xa / self.league_mean_xg),
                real_pts_avg=float(s['pts'] / ws),
                matches_played=int(s['count']),
                quality=float(quality),
                stability=float(stability),
                hfa=self.home_advantage
            )
        logging.info(f"✅ Titanium Brain: Рейтинги обновлены для {len(self.ratings)} команд.")

    def load_injuries_real(self):
        if not os.path.exists(self.injuries_path):
            return
        try:
            with open(self.injuries_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                elements = data.get("elements", [])

                if "teams" in data:
                    for t in data["teams"]:
                        team_id = t.get("id")
                        team_name = t.get("name")
                        if team_id is not None and team_name:
                            self.fpl_id_map[int(team_id)] = team_name.lower()

            eps = [
                float(p.get("ep_this") or 0)
                for p in elements
                if float(p.get("ep_this") or 0) > 0
            ]
            typical_ep = np.percentile(eps, 90) if eps else 5.0

            for p in elements:
                if p.get("status") == "a":
                    continue
                t_id = int(p.get("team", 0)) if p.get("team") else None
                t_name = self.fpl_id_map.get(t_id)
                if not t_name:
                    continue
                t_c = self.canonicalize(t_name, True)
                if t_c not in self.ratings:
                    continue
                if t_c not in self.injuries:
                    self.injuries[t_c] = {"att": 0.0, "def": 0.0}

                ep_val = float(p.get("ep_this") or 0)
                impact = ep_val / typical_ep
                p_type = p.get("element_type")

                if p_type == 1:
                    self.injuries[t_c]["def"] += impact * 0.40
                elif p_type == 2:
                    self.injuries[t_c]["def"] += impact * 0.20
                elif p_type == 3:
                    self.injuries[t_c]["att"] += impact * 0.15
                    self.injuries[t_c]["def"] += impact * 0.05
                elif p_type == 4:
                    self.injuries[t_c]["att"] += impact * 0.35

            logging.info(f"✅ Injury impact calculated for {len(self.injuries)} teams.")
        except Exception as e:
            logging.error(f"Injury sync error: {e}")

    def calculate_probs(self, home: str, away: str) -> Optional[dict]:
        hc, ac = self.canonicalize(home), self.canonicalize(away)
        if hc not in self.ratings or ac not in self.ratings:
            return None

        rh, ra = self.ratings[hc], self.ratings[ac]
        h_fatigue = self._get_fatigue_factor(hc)
        a_fatigue = self._get_fatigue_factor(ac)

        h_inj = self.injuries.get(hc, {"att": 0.0, "def": 0.0})
        a_inj = self.injuries.get(ac, {"att": 0.0, "def": 0.0})

        max_inj = self.cfg["max_injury_frac"]
        h_att_mod = max(1 - max_inj, 1.0 - h_inj["att"])
        h_def_mod = min(1 + max_inj, 1.0 + h_inj["def"])
        a_att_mod = max(1 - max_inj, 1.0 - a_inj["att"])
        a_def_mod = min(1 + max_inj, 1.0 + a_inj["def"])

        lh = (
            (rh.att_norm * ra.def_norm * 0.8 + 0.2)
            * self.league_mean_xg
            * rh.hfa
            * h_fatigue
            * h_att_mod
            * a_def_mod
        )
        la = (
            (ra.att_norm * rh.def_norm * 0.8 + 0.2)
            * self.league_mean_xg
            * a_fatigue
            * a_att_mod
            * h_def_mod
        )

        lh *= 1.20 
        la *= 1.20

        max_g = int(max(14, np.ceil(max(lh, la) + 6 * np.sqrt(max(lh, la)))))
        m = np.outer(
            poisson.pmf(np.arange(max_g), lh), poisson.pmf(np.arange(max_g), la)
        )

        rho = self.cfg["rho"]
        dc = {
            (0, 0): 1 - lh * la * rho,
            (1, 0): 1 + la * rho,
            (0, 1): 1 + lh * rho,
            (1, 1): 1 - rho,
        }
        for (i, j), v in dc.items():
            if i < max_g and j < max_g:
                m[i, j] *= np.clip(v, 0.75, 1.25)

        s = m.sum()
        if s < 1e-6:
            p1 = 1 - skellam.cdf(0, lh, la)
            px = skellam.pmf(0, lh, la)
            p2 = skellam.cdf(-1, lh, la)
            tb25 = 1 - poisson.cdf(2, lh + la)
            btts = (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
            top_scores = []
        else:
            m /= s
            # П1: Сумма всех ячеек, где голы хозяев (строки) > голов гостей (столбцы)
            p1 = float(np.sum(np.tril(m, -1))) 
            
            # Х: Только диагональ (счета 0:0, 1:1, 2:2 и т.д.)
            px = float(np.sum(np.diag(m)))
            
            # П2: Сумма всех ячеек, где голы гостей (столбцы) > голов хозяев (строки)
            p2 = float(np.sum(np.triu(m, 1)))
            
            # Остальное оставляем как есть
            btts = float(np.sum(m[1:, 1:]))
            goals_sum_matrix = np.add.outer(np.arange(max_g), np.arange(max_g))
            tb25 = 1 - float(m[goals_sum_matrix <= 2].sum())
            score_list = [
                (f"{hg}:{ag}", float(m[hg, ag])) for hg in range(5) for ag in range(5)
            ]
            top_scores = sorted(score_list, key=lambda x: x[1], reverse=True)[:3]

        return {
            "P1": float(p1),
            "X": float(px),
            "P2": float(p2),
            "TB25": float(tb25),
            "BTTS": float(btts),
            "lambdas": (float(lh), float(la)),
            "top_scores": top_scores,
        }
    
    def calculate_margin(self, odds: list) -> float:
        """Внутренний метод класса для расчета маржи"""
        return calculate_margin_global(odds)

    def run_cli(self):
        if not os.path.exists(self.upcoming_path):
            print(f"❌ Файл не найден: {self.upcoming_path}")
            return

        try:
            with open(self.upcoming_path, "r", encoding="utf-8") as f:
                fixtures = json.load(f)
        except Exception as e:
            print(f"❌ Ошибка JSON: {e}")
            return

        now_ms = datetime.now(timezone.utc).timestamp() * 1000
        MSK_OFFSET = 3 * 3600 * 1000
        fixtures.sort(key=lambda x: x.get("timestamp", 0))

        current_gw = None
        for game in fixtures:
            ts = game.get("timestamp", 0)
            if ts + (115 * 60 * 1000) > now_ms:
                current_gw = game.get("gameweek")
                break
        if not current_gw and fixtures:
            current_gw = fixtures[-1].get("gameweek")

        upcoming = [f for f in fixtures if f.get("gameweek") == current_gw]

        print(f"\n{'='*55}")
        print(f"💎 DIAMOND ROCKET V5.3 TITANIUM")
        print(f"{'='*55}")
        print(f"🏆 АКТУАЛЬНЫЙ ТУР: {current_gw} из 38")
        print(f"{'='*55}")

        if not upcoming:
            print("📅 Матчей в текущем туре не найдено.")
            return

        for i, game in enumerate(upcoming):
            try:
                h = game.get("homeTeam", {}).get("name", "Unknown")
                a = game.get("awayTeam", {}).get("name", "Unknown")
                ts_msk = game.get("timestamp", 0) + MSK_OFFSET
                dt_msk = datetime.fromtimestamp(ts_msk / 1000, tz=timezone.utc)
                print(f"[{i}] {dt_msk.strftime('%a %d %b, %H:%M')} МСК | {h} vs {a}")
            except Exception:
                continue

        while True:
            cmd = input(f"\n🆔 ID матча (или 'q' для выхода): ").strip()
            if cmd.lower() == "q":
                break

            try:
                idx = int(cmd)
                target = upcoming[idx]
                h, a = target["homeTeam"]["name"], target["awayTeam"]["name"]

                res = self.calculate_probs(h, a)
                if not res:
                    print(f"⚠️ Команды нет в рейтингах!")
                    print(f"Доступные: {list(self.ratings.keys())[:10]}...")
                    continue

                print(f"\n📈 Введите коэффициенты (Enter для пропуска) {h} - {a}:")
                bk_p1 = self.safe_float("КФ на П1: ", can_skip=True)
                bk_px = self.safe_float("КФ на Х:  ", can_skip=True)
                bk_p2 = self.safe_float("КФ на П2: ", can_skip=True)
                bk_tb = self.safe_float("КФ на ТБ 2.5: ", can_skip=True)
                bk_btts = self.safe_float("КФ на ОЗ (Да): ", can_skip=True)

                # Считаем маржу только по тем исходам, которые ввели (обычно П1-Х-П2)
                margin = self.calculate_margin([bk_p1, bk_px, bk_p2])

                # Формируем список возможных исходов
                all_possible = [
                    ("П1", res["P1"], bk_p1),
                    ("X", res["X"], bk_px),
                    ("П2", res["P2"], bk_p2),
                    ("ТБ 2.5", res["TB25"], bk_tb),
                    ("Обе забьют", res["BTTS"], bk_btts),
                ]
                # Оставляем только те, где введен коэффициент
                outcomes = [opt for opt in all_possible if opt[2] is not None]

                if not outcomes:
                    print(
                        "\n⚠️ Ни один коэффициент не введен. Расчет Value невозможен."
                    )
                    best_bet = ("Нет данных", 0.001, 1.0)  # Заглушка, чтобы не падало
                    value = -1.0
                else:
                    # Находим лучший исход
                    best_bet_raw = max(outcomes, key=lambda x: (x[1] * x[2]) - 1)
                    # Распаковываем для удобства
                    best_label, best_prob, best_odd = best_bet_raw
                    
                    # --- ФИЛЬТР ЗДРАВОГО СМЫСЛА (ANTI-MIRACLE FILTER) ---
                    # Используем вероятности из res["P1"] и res["P2"]
                    prob_h = res["P1"]
                    prob_a = res["P2"]
                    
                    # Если одна команда в 2 раза сильнее другой по вероятности, 
                    # блокируем ставку на "чудо" аутсайдера
                    if prob_a > prob_h * 1.4 and best_label == "П1":
                        # Если Арсенал сильнее, но кэф тянет на Лидс — меняем на П2 (Арсенал)
                        # Ищем П2 в списке outcomes, чтобы взять его данные
                        for opt in outcomes:
                            if opt[0] == "П2":
                                best_label, best_prob, best_odd = opt
                        logging.info("🛡 Сработал фильтр класса: ставка переведена на фаворита.")
                    elif prob_h > prob_a * 2 and best_label == "П2":
                        for opt in outcomes:
                            if opt[0] == "П1":
                                best_label, best_prob, best_odd = opt
                    
                    # Пересобираем итоговый выбор
                    best_bet = (best_label, best_prob, best_odd)
                    value = (best_prob * best_odd) - 1

                # Определение статуса и иконок
                if value > 0.40:
                    status = "🚨 АНОМАЛИЯ (Проверьте данные!)"
                elif value > 0.10:
                    status = "🔥 ВЫГОДНО (Strong Value)"
                elif value >= 0:
                    status = "✅ ВАЛУЙ (Риск)"
                else:
                    status = "❌ НЕ ВЫГОДНО"

                # Безопасный расчет честного кэфа
                fair_odd = 1 / best_bet[1] if best_bet[1] > 0 else 0

                print(f"\n{'='*55}")
                print(f"💎 ВЕРДИКТ DIAMOND: {h} — {a}")
                print(f"{'='*55}")
                print(f"🤖 {self.brain.say_hello()}")
                print(f"\n📊 Статус: {status}")
                print(f"   ROI (Перевес): {value * 100:+.1f}%")
                print(f"-" * 55)

                # Блок 1: Основной выбор (теперь выводится всегда)
                print(f"1. 🎯 Основной выбор: [{best_bet[0]}]")
                print(
                    f"   🏦 БК Коэф: {best_bet[2]:<6.2f} | 🧠 Честный Коэф: {fair_odd:<6.2f}"
                )

                # Блок 2: Ожидаемые голы
                print(f"2. ⚽ xG (Ожидаемые голы):")
                print(f"   {h}: {res['lambdas'][0]:.2f}")
                print(f"   {a}: {res['lambdas'][1]:.2f}")

                # Блок 3: Вероятности
                print(f"3. 🎲 Вероятности:")
                print(f"   ТМ 2.5: {(1 - res['TB25']) * 100:>5.1f}%")
                print(f"   ТБ 2.5: {res['TB25'] * 100:>5.1f}%")
                print(f"   ОЗ(Да): {res['BTTS'] * 100:>5.1f}%")

                # Блок 4: Точный счет
                scores_str = ", ".join(
                    [f"{s[0]} ({s[1] * 100:.1f}%)" for s in res["top_scores"]]
                )
                print(f"4. 🔮 Точный счет: {scores_str}")

                # --- ДОБАВЬ ЭТОТ БЛОК ДЛЯ СОВЕТОВ ---
                advice = ""
                if value > 0.40:
                    advice = "⚠️ ВЫСОКИЙ РИСК: Коэффициент аномально выгоден. Возможно, рынок не знает о чем-то важном (травмы, ротация)."
                elif value > 0.10:
                    advice = f"✅ ХОРОШИЙ ВАРИАНТ: Математика на стороне [{best_bet[0]}]. Дистанция должна вернуть прибыль."
                else:
                    advice = "☕ ОСТОРОЖНО: Перевес минимален. Лучше пропустить этот матч или зайти символической суммой."

                if res['TB25'] < 0.40 and res['BTTS'] < 0.40:
                    advice += "\n🛡 ОЖИДАЕТСЯ ЗАКРЫТАЯ ИГРА: Модель прогнозирует низкую результативность."
                
                # Выводим совет в консоль
                print(f"💡 СОВЕТ: {advice}")

                # Расчет ставки (Келли)
                bet_amt = 0
                if value > 0:
                    odds = best_bet[2]
                    # Если перевес аномальный (>40%), режем ставку в 2 раза для безопасности
                    safety_factor = 0.5 if value > 0.40 else 1.0

                    kelly_pct = (value / (odds - 1)) * self.cfg.get(
                        "kelly_fraction", 0.1
                    )
                    final_pct = np.clip(
                        kelly_pct, 0, self.cfg.get("max_bank_pct", 0.05)
                    )
                    bet_amt = self.bankroll * final_pct * safety_factor

                print(f"-" * 55)
                print(f"💰 РЕКОМЕНДУЕМАЯ СТАВКА: {bet_amt:.0f} руб.")
                print(f"🏁 МАРЖА БК: {margin * 100:.1f}%")

                self.save_prediction(
                    {
                        "game": f"{h} vs {a}",
                        "probs": res,
                        "bk_odds": {
                            "P1": bk_p1,
                            "X": bk_px,
                            "P2": bk_p2,
                            "TB25": bk_tb,
                            "BTTS": bk_btts,
                        },
                        "value": value,
                        "bet_suggestion": best_bet[0],
                        "stake": bet_amt,
                    }
                )
                print(f"✅ Прогноз сохранен в историю.")
                print(f"{'='*55}\n")

            except ValueError:
                print("⚠️ Введите корректный номер матча.")
            except IndexError:
                print("⚠️ Матч с таким номером не найден.")
            except Exception:
                import traceback

                print("❌ ОШИБКА:")
                print(traceback.format_exc())

    def calculate_margin(self, odds: List[Optional[float]]) -> float:
        valid_odds = [o for o in odds if o is not None and o > 0]
        if not valid_odds:
            return 0.0
        return sum(1.0 / o for o in valid_odds) - 1.0

    def save_prediction(self, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        history = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass

        teams = data["game"].split(" vs ")
        clean_key = f"{self.canonicalize(teams[0], True)} vs {self.canonicalize(teams[1], True)}"

        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "game": clean_key,
                "display_name": data["game"],
                "predictions": data["probs"],
                "bk_odds": data["bk_odds"],
                "value": data["value"],
                "bet_suggestion": data["bet_suggestion"],
                "stake": data["stake"],
            }
        )

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def settle_bets(self):
        if not os.path.exists(self.history_path):
            return

        if not os.path.exists(self.odds_dir):
            return

        results = {}
        files = [f for f in os.listdir(self.odds_dir) if f.endswith(".csv")]
        
        # Исправленный блок: загружаем данные из файлов
        for f in files:
            try:
                path = os.path.join(self.odds_dir, f)
                df = pd.read_csv(path) # Теперь df определен
                for _, row in df.iterrows():
                    if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                        continue
                
                    key = f"{self.canonicalize(row['HomeTeam'], True)} vs {self.canonicalize(row['AwayTeam'], True)}"
                    results[key] = {
                        "hg": int(row["FTHG"]), 
                        "ag": int(row["FTAG"])
                    }
            except Exception as e:
                logging.warning(f"Ошибка при обработке файла {f} для расчета ставок: {e}")
                continue

        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return

        updated = False
        for entry in history:
            if "pnl" not in entry:
                game_key = entry["game"]
                if game_key in results:
                    hg, ag = results[game_key]["hg"], results[game_key]["ag"]
                    bet = entry["bet_suggestion"]
                    odds_map = {
                        "П1": "P1",
                        "П2": "P2",
                        "X": "X",
                        "ТБ 2.5": "TB25",
                        "Обе забьют": "BTTS",
                    }
                    odds = entry["bk_odds"].get(odds_map.get(bet, "P1"), 0)

                    win = False
                    if bet == "П1" and hg > ag:
                        win = True
                    elif bet == "П2" and ag > hg:
                        win = True
                    elif bet == "X" and hg == ag:
                        win = True
                    elif bet == "ТБ 2.5" and (hg + ag) > 2.5:
                        win = True
                    elif bet == "Обе забьют" and hg > 0 and ag > 0:
                        win = True

                    stake = entry.get("stake", 0)
                    entry["pnl"] = (stake * odds - stake) if win else -stake
                    entry["score"] = f"{hg}:{ag}"
                    updated = True
                    logging.info(
                        f"✅ Settled: {game_key} | {entry['score']} | PnL: {entry['pnl']:.2f}"
                    )

        if updated:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(BASE_DIR, "data", "premierleague")
    MatchPredictor(DATA).run_cli()
