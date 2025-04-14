import os
import cv2
import numpy as np
# Устанавливаем Agg бэкенд для matplotlib перед импортом pyplot
import matplotlib
matplotlib.use('Agg')  # Не-интерактивный бэкенд без GUI
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union

class SSLModel(nn.Module):
    """
    Модель самоконтролируемого обучения для извлечения признаков из изображений.
    Основана на ResNet50 с проекционной головой.
    """
    def __init__(self, feature_dim=128, use_pretrained=True):
        super(SSLModel, self).__init__()
        
        # Загружаем базовую модель ResNet50
        resnet = models.resnet50(pretrained=use_pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # удаляем FC слой
        
        # Проекционная голова для преобразования признаков
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        # Прямой проход через энкодер
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        
        # Проекция признаков в пространство меньшей размерности
        z = self.projector(features)
        
        # Нормализация для косинусного сходства
        z = F.normalize(z, dim=1)
        
        return z

class ObjectDetector:
    """
    Класс для обнаружения объектов на видео с использованием
    самоконтролируемого обучения.
    """
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Загружаем модель
        self.model = SSLModel(feature_dim=128)
        
        # Пытаемся загрузить веса модели
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using pretrained ResNet50 without fine-tuning")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Определяем преобразования для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_path: str) -> torch.Tensor:
        """
        Извлекает признаки из изображения.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Тензор признаков
        """
        try:
            # Загружаем и преобразуем изображение
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Извлекаем признаки
            with torch.no_grad():
                features = self.model(image_tensor)
                
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_features_from_multiple_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Извлекает признаки из нескольких изображений и возвращает их среднее значение.
        
        Args:
            image_paths: Список путей к изображениям
            
        Returns:
            Тензор признаков (среднее значение всех изображений)
        """
        if not image_paths:
            raise ValueError("No image paths provided")
        
        all_features = []
        
        for path in image_paths:
            features = self.extract_features(path)
            if features is not None:
                all_features.append(features)
        
        if not all_features:
            raise ValueError("Failed to extract features from any of the provided images")
        
        # Объединяем все признаки и вычисляем среднее значение
        combined_features = torch.cat(all_features, dim=0)
        avg_features = torch.mean(combined_features, dim=0, keepdim=True)
        
        # Нормализуем усредненные признаки
        normalized_features = F.normalize(avg_features, dim=1)
        
        return normalized_features
    
    def calculate_similarity(self, target_features: torch.Tensor, frame_features: torch.Tensor) -> float:
        """
        Вычисляет косинусное сходство между признаками целевого объекта и кадра.
        
        Args:
            target_features: Признаки целевого объекта
            frame_features: Признаки кадра
            
        Returns:
            Значение сходства [0, 1]
        """
        # Косинусное сходство
        similarity = F.cosine_similarity(target_features, frame_features)
        
        return similarity.item()
    
    def process_video_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Обрабатывает кадр видео и извлекает признаки.
        
        Args:
            frame: Кадр видео в формате numpy array (BGR)
            
        Returns:
            Тензор признаков
        """
        # Конвертируем BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Преобразуем в PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Применяем трансформации
        frame_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Извлекаем признаки
        with torch.no_grad():
            features = self.model(frame_tensor)
            
        return features
    
    def visualize_results(self, 
                          video_path: str, 
                          target_image_path: Union[str, List[str]], 
                          output_image_path: Optional[str] = None,
                          threshold: float = 0.85,
                          sample_rate: int = 10) -> Tuple[List[int], List[float], List[int]]:
        """
        Анализирует видео на наличие целевого объекта и визуализирует результаты.
        
        Args:
            video_path: Путь к видео
            target_image_path: Путь к изображению целевого объекта или список путей
            output_image_path: Путь для сохранения графика результатов
            threshold: Порог сходства для обнаружения объекта
            sample_rate: Частота выборки кадров (каждый N-й кадр)
            
        Returns:
            Кортеж из (номера кадров, значения сходства, номера совпадающих кадров)
        """
        # Извлекаем признаки целевого объекта (одного или нескольких)
        if isinstance(target_image_path, list):
            target_features = self.extract_features_from_multiple_images(target_image_path)
            print(f"Using {len(target_image_path)} target images for analysis")
        else:
            target_features = self.extract_features(target_image_path)
            print(f"Using a single target image for analysis")
        
        if target_features is None:
            raise ValueError(f"Failed to extract features from target image(s)")
        
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Списки для хранения результатов
        frames = []
        similarities = []
        matching_frames = []
        
        # Обрабатываем видео
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Обрабатываем только каждый N-й кадр
            if frame_idx % sample_rate == 0:
                # Извлекаем признаки из кадра
                frame_features = self.process_video_frame(frame)
                
                # Вычисляем сходство
                similarity = self.calculate_similarity(target_features, frame_features)
                
                # Сохраняем результаты
                frames.append(frame_idx)
                similarities.append(similarity)
                
                # Проверяем, превышен ли порог
                if similarity >= threshold:
                    matching_frames.append(frame_idx)
                    
                print(f"Frame {frame_idx}/{total_frames}: similarity = {similarity:.4f}")
            
            frame_idx += 1
        
        cap.release()
        
        # Визуализируем результаты
        if output_image_path:
            self._plot_results(frames, similarities, matching_frames, threshold, fps, output_image_path)
            
        return frames, similarities, matching_frames
    
    def _plot_results(self, 
                      frames: List[int], 
                      similarities: List[float], 
                      matching_frames: List[int],
                      threshold: float,
                      fps: float,
                      output_image_path: str):
        """
        Создает график результатов анализа.
        
        Args:
            frames: Номера кадров
            similarities: Значения сходства
            matching_frames: Номера совпадающих кадров
            threshold: Порог сходства
            fps: Частота кадров видео
            output_image_path: Путь для сохранения графика
        """
        # Преобразуем номера кадров в секунды
        time_points = [frame / fps for frame in frames]
        
        plt.figure(figsize=(12, 6))
        
        # График сходства
        plt.plot(time_points, similarities, 'b-', label='Сходство')
        
        # Отмечаем кадры, превышающие порог
        if matching_frames:
            matching_times = [frame / fps for frame in matching_frames]
            matching_similarities = [similarities[frames.index(frame)] for frame in matching_frames]
            plt.plot(matching_times, matching_similarities, 'ro', label='Обнаружения')
        
        # Добавляем линию порога
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Порог ({threshold})')
        
        # Оформление графика
        plt.xlabel('Время (секунды)')
        plt.ylabel('Сходство')
        plt.title('Анализ сходства объекта на видео')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        
        # Сохраняем график
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results plot saved to {output_image_path}")

    def analyze_video(self, 
                      video_path: str, 
                      target_image_paths: List[str],
                      threshold: float = 0.85,
                      sample_rate: int = 10) -> Dict:
        """
        Комплексный анализ видео для поиска нескольких целевых объектов.
        
        Args:
            video_path: Путь к видео
            target_image_paths: Список списков путей к изображениям целевых объектов
                               (каждый внутренний список - один объект)
            threshold: Порог сходства для обнаружения объекта
            sample_rate: Частота выборки кадров (каждый N-й кадр)
            
        Returns:
            Словарь с результатами анализа
        """
        # Проверяем входные данные
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Открываем видео для получения параметров
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        cap.release()
        
        # Результаты для каждого объекта
        results = {}
        
        # Анализируем каждый объект
        for idx, image_paths in enumerate(target_image_paths):
            product_id = f"product_{idx}"
            
            # Проверяем пути к изображениям
            valid_paths = []
            for path in image_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    print(f"Warning: Image file not found: {path}")
            
            if not valid_paths:
                print(f"Error: No valid image files found for {product_id}")
                continue
            
            print(f"Analyzing video for {product_id} using {len(valid_paths)} images...")
            
            try:
                # Анализируем видео для текущего объекта
                frames, similarities, matching_frames = self.visualize_results(
                    video_path=video_path,
                    target_image_path=valid_paths,
                    threshold=threshold,
                    sample_rate=sample_rate
                )
                
                # Вычисляем ключевые метрики
                if similarities:
                    max_similarity = max(similarities)
                    avg_similarity = sum(similarities) / len(similarities)
                    
                    # Вычисляем время появления объекта в видео
                    if matching_frames:
                        appearance_time = len(matching_frames) * sample_rate / fps
                        appearance_percentage = appearance_time / duration * 100
                    else:
                        appearance_time = 0
                        appearance_percentage = 0
                    
                    # Сохраняем результаты
                    results[product_id] = {
                        'frames': frames,
                        'similarities': similarities,
                        'matching_frames': matching_frames,
                        'max_similarity': max_similarity,
                        'avg_similarity': avg_similarity,
                        'appearance_time': appearance_time,
                        'appearance_percentage': appearance_percentage,
                        'status': 'success'
                    }
                else:
                    results[product_id] = {
                        'status': 'error',
                        'message': 'No similarity data generated'
                    }
            
            except Exception as e:
                results[product_id] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # Добавляем общую информацию
        results['metadata'] = {
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'threshold': threshold,
            'sample_rate': sample_rate
        }
        
        return results