#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能建筑综合布线系统优化案例
Smart Building Integrated Cabling System Optimization

作者：智子 (Sophon)
日期：2026-04-01
描述：实现智能建筑中的综合布线系统优化，包括：
      - 多层建筑建模 (楼层/房间/设备)
      - 6 大子系统 (工作区/水平/垂直/设备间/进线/管理)
      - 电缆路径优化 (MST + 约束满足)
      - 电磁干扰规避
      - 未来扩展预留

参考标准:
      - GB 50311-2016 综合布线系统工程设计规范
      - TIA/EIA-568 商业建筑电信布线标准
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import heapq
from collections import defaultdict


# ============================================================================
# 数据结构定义
# ============================================================================

class CableCategory(Enum):
    """电缆类别枚举"""
    CAT5E = "Cat5e"           # 超五类 (1Gbps, 100m)
    CAT6 = "Cat6"             # 六类 (10Gbps, 55m)
    CAT6A = "Cat6A"           # 超六类 (10Gbps, 100m)
    CAT7 = "Cat7"             # 七类 (10Gbps+, 100m)
    FIBER_OM3 = "Fiber-OM3"   # 多模光纤 (40/100G, 300m)
    FIBER_OS2 = "Fiber-OS2"   # 单模光纤 (100G+, 10km+)
    COAXIAL = "Coaxial"       # 同轴电缆 (视频/RF)
    POWER = "Power"           # 电源线


class RoomType(Enum):
    """房间类型枚举"""
    OFFICE = "办公室"
    CONFERENCE = "会议室"
    SERVER_ROOM = "机房"
    LOBBY = "大厅"
    RESTROOM = "卫生间"
    STAIRWELL = "楼梯间"
    ELEVATOR = "电梯间"
    STORAGE = "储藏室"


class InterferenceSource(Enum):
    """干扰源类型"""
    POWER_LINE = "电源线"       # 强电磁干扰
    ELEVATOR = "电梯"           # 电机干扰
    HVAC = "空调系统"           # 变频干扰
    MICROWAVE = "微波炉"        # 高频干扰
    RADIO = "无线电设备"        # RF 干扰


@dataclass
class Position3D:
    """3D 空间位置"""
    x: float  # X 坐标 (米)
    y: float  # Y 坐标 (米)
    z: int    # 楼层 (1-based)
    
    def distance_to(self, other: 'Position3D') -> float:
        """计算 3D 欧氏距离"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = (self.z - other.z) * 3.5  # 层高 3.5 米
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def horizontal_distance(self, other: 'Position3D') -> float:
        """计算水平距离 (同层或投影)"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Room:
    """房间定义"""
    id: str
    room_type: RoomType
    position: Position3D
    area: float  # 面积 (平方米)
    work_areas: int = 1  # 工作区数量
    bandwidth_requirement: float = 1.0  # 带宽需求 (Gbps)
    special_requirements: List[str] = field(default_factory=list)


@dataclass
class Device:
    """网络设备"""
    id: str
    name: str
    device_type: str  # "AP", "Camera", "Phone", "PC", "Sensor"
    position: Position3D
    room_id: str
    bandwidth: float  # 带宽需求 (Mbps)
    cable_type: CableCategory
    power_over_ethernet: bool = False  # 是否 PoE


@dataclass
class TelecommunicationsRoom:
    """电信间 (配线间)"""
    id: str
    floor: int
    position: Position3D
    serves_floors: List[int]  # 服务的楼层范围
    devices: List[Device] = field(default_factory=list)
    uplink_to: Optional[str] = None  # 上级电信间


@dataclass
class EquipmentRoom:
    """设备间 (主配线间)"""
    id: str
    position: Position3D
    telecom_rooms: List[str] = field(default_factory=list)  # 下属电信间
    external_connections: List[str] = field(default_factory=list)


@dataclass
class InterferenceZone:
    """电磁干扰区域"""
    id: str
    source_type: InterferenceSource
    center: Position3D
    radius: float  # 影响半径 (米)
    severity: float  # 严重程度 (0-1)
    
    def affects_position(self, pos: Position3D) -> bool:
        """检查是否影响某位置"""
        return self.center.horizontal_distance(pos) <= self.radius


@dataclass
class CablePath:
    """电缆路径"""
    id: str
    from_node: str
    to_node: str
    cable_type: CableCategory
    length: float
    route: List[Position3D]
    cost: float
    interference_score: float = 0.0  # 干扰评分 (0-1, 越低越好)


# ============================================================================
# 智能建筑模型
# ============================================================================

class SmartBuilding:
    """智能建筑模型"""
    
    def __init__(self, name: str, floors: int, floor_area: Tuple[float, float]):
        self.name = name
        self.floors = floors
        self.floor_width = floor_area[0]
        self.floor_depth = floor_area[1]
        
        self.rooms: Dict[str, Room] = {}
        self.devices: Dict[str, Device] = {}
        self.telecom_rooms: Dict[str, TelecommunicationsRoom] = {}
        self.equipment_room: Optional[EquipmentRoom] = None
        self.interference_zones: List[InterferenceZone] = []
        self.cable_paths: List[CablePath] = []
        
    def add_room(self, room: Room):
        """添加房间"""
        self.rooms[room.id] = room
        
    def add_device(self, device: Device):
        """添加设备"""
        self.devices[device.id] = device
        
    def add_telecom_room(self, tr: TelecommunicationsRoom):
        """添加电信间"""
        self.telecom_rooms[tr.id] = tr
        
    def set_equipment_room(self, er: EquipmentRoom):
        """设置设备间"""
        self.equipment_room = er
        
    def add_interference_zone(self, zone: InterferenceZone):
        """添加干扰区域"""
        self.interference_zones.append(zone)
        
    def get_devices_on_floor(self, floor: int) -> List[Device]:
        """获取某楼层的所有设备"""
        return [d for d in self.devices.values() if d.position.z == floor]
    
    def get_nearest_telecom_room(self, position: Position3D) -> TelecommunicationsRoom:
        """获取最近的电信间"""
        if not self.telecom_rooms:
            raise ValueError("没有电信间")
        
        nearest = None
        min_dist = float('inf')
        
        for tr in self.telecom_rooms.values():
            dist = position.distance_to(tr.position)
            if dist < min_dist:
                min_dist = dist
                nearest = tr
        
        return nearest


# ============================================================================
# 电缆路由优化器
# ============================================================================

class CableRoutingOptimizer:
    """电缆路由优化器"""
    
    def __init__(self, building: SmartBuilding):
        self.building = building
        self.cable_costs = {
            CableCategory.CAT5E: 3.0,      # 元/米
            CableCategory.CAT6: 5.0,
            CableCategory.CAT6A: 8.0,
            CableCategory.CAT7: 12.0,
            CableCategory.FIBER_OM3: 25.0,
            CableCategory.FIBER_OS2: 40.0,
            CableCategory.COAXIAL: 6.0,
            CableCategory.POWER: 15.0,
        }
        
    def calculate_cable_type(self, device: Device, distance: float) -> CableCategory:
        """根据带宽和距离选择合适的电缆类型"""
        bandwidth = device.bandwidth  # Mbps
        
        if bandwidth <= 1000 and distance <= 100:
            return CableCategory.CAT5E
        elif bandwidth <= 10000 and distance <= 55:
            return CableCategory.CAT6
        elif bandwidth <= 10000 and distance <= 100:
            return CableCategory.CAT6A
        elif bandwidth > 10000:
            if distance <= 300:
                return CableCategory.FIBER_OM3
            else:
                return CableCategory.FIBER_OS2
        else:
            return device.cable_type
    
    def calculate_interference_score(self, route: List[Position3D]) -> float:
        """计算路径的干扰评分"""
        if not self.building.interference_zones:
            return 0.0
        
        total_score = 0.0
        
        for pos in route:
            for zone in self.building.interference_zones:
                if zone.affects_position(pos):
                    distance = zone.center.horizontal_distance(pos)
                    normalized_dist = distance / zone.radius
                    score = zone.severity * (1 - normalized_dist)
                    total_score += score
        
        return total_score / max(len(route), 1)
    
    def route_device_to_telecom(self, device: Device) -> CablePath:
        """为设备路由到电信间"""
        telecom = self.building.get_nearest_telecom_room(device.position)
        
        # 简单路由：直线路径
        route = [device.position, telecom.position]
        length = device.position.distance_to(telecom.position)
        
        # 考虑垂直布线 (通过线井)
        if device.position.z != telecom.position.z:
            # 添加线井转折点
            riser_pos = Position3D(
                x=telecom.position.x,
                y=telecom.position.y,
                z=device.position.z
            )
            route = [device.position, riser_pos, telecom.position]
            length = (device.position.horizontal_distance(riser_pos) + 
                     riser_pos.distance_to(telecom.position))
        
        cable_type = self.calculate_cable_type(device, length)
        cost = length * self.cable_costs[cable_type]
        interference = self.calculate_interference_score(route)
        
        return CablePath(
            id=f"path_{device.id}_to_{telecom.id}",
            from_node=device.id,
            to_node=telecom.id,
            cable_type=cable_type,
            length=length,
            route=route,
            cost=cost,
            interference_score=interference
        )
    
    def route_telecom_to_equipment(self, telecom: TelecommunicationsRoom) -> CablePath:
        """为电信间路由到设备间"""
        if not self.building.equipment_room:
            raise ValueError("没有设备间")
        
        equipment = self.building.equipment_room
        
        # 垂直干线通过线井
        route = [telecom.position, equipment.position]
        length = telecom.position.distance_to(equipment.position)
        
        # 干线使用光纤
        cable_type = CableCategory.FIBER_OM3
        cost = length * self.cable_costs[cable_type]
        interference = self.calculate_interference_score(route)
        
        return CablePath(
            id=f"backbone_{telecom.id}_to_{equipment.id}",
            from_node=telecom.id,
            to_node=equipment.id,
            cable_type=cable_type,
            length=length,
            route=route,
            cost=cost,
            interference_score=interference
        )
    
    def optimize_all_routes(self) -> Dict[str, float]:
        """优化所有路由"""
        stats = {
            'total_cable_length': 0.0,
            'total_cost': 0.0,
            'horizontal_cables': 0,
            'backbone_cables': 0,
            'avg_interference': 0.0,
        }
        
        interference_sum = 0.0
        path_count = 0
        
        # 1. 水平布线 (设备到电信间)
        for device in self.building.devices.values():
            path = self.route_device_to_telecom(device)
            self.building.cable_paths.append(path)
            
            stats['total_cable_length'] += path.length
            stats['total_cost'] += path.cost
            stats['horizontal_cables'] += 1
            interference_sum += path.interference_score
            path_count += 1
        
        # 2. 垂直干线 (电信间到设备间)
        for telecom in self.building.telecom_rooms.values():
            path = self.route_telecom_to_equipment(telecom)
            self.building.cable_paths.append(path)
            
            stats['total_cable_length'] += path.length
            stats['total_cost'] += path.cost
            stats['backbone_cables'] += 1
            interference_sum += path.interference_score
            path_count += 1
        
        if path_count > 0:
            stats['avg_interference'] = interference_sum / path_count
        
        return stats


# ============================================================================
# 可视化
# ============================================================================

class SmartBuildingVisualizer:
    """智能建筑可视化"""
    
    def __init__(self, building: SmartBuilding, optimizer: CableRoutingOptimizer):
        self.building = building
        self.optimizer = optimizer
        self.cable_colors = {
            CableCategory.CAT5E: '#90EE90',
            CableCategory.CAT6: '#32CD32',
            CableCategory.CAT6A: '#228B22',
            CableCategory.CAT7: '#006400',
            CableCategory.FIBER_OM3: '#FFD700',
            CableCategory.FIBER_OS2: '#FFA500',
            CableCategory.COAXIAL: '#4169E1',
            CableCategory.POWER: '#DC143C',
        }
        
    def plot_floor_plan(self, floor: int, save_path: Optional[str] = None):
        """绘制某楼层平面图"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制楼层边界
        ax.add_patch(plt.Rectangle((0, 0), self.building.floor_width, 
                                   self.building.floor_depth, 
                                   fill=False, linewidth=3, edgecolor='black'))
        
        # 绘制房间
        for room in self.building.rooms.values():
            if room.position.z == floor:
                color = self._get_room_color(room.room_type)
                ax.add_patch(plt.Rectangle(
                    (room.position.x - 2, room.position.y - 2),
                    4, 4,
                    fill=True, alpha=0.3, color=color,
                    label=room.room_type.value
                ))
                ax.text(room.position.x, room.position.y, room.id,
                       ha='center', va='center', fontsize=8)
        
        # 绘制设备
        for device in self.building.get_devices_on_floor(floor):
            ax.plot(device.position.x, device.position.y, 'o', 
                   color=self.cable_colors[device.cable_type],
                   markersize=10, label=device.device_type)
        
        # 绘制电信间
        for tr in self.building.telecom_rooms.values():
            if tr.position.z == floor or floor in tr.serves_floors:
                ax.plot(tr.position.x, tr.position.y, 's', 
                       color='blue', markersize=15, label='电信间')
        
        # 绘制干扰区域
        for zone in self.building.interference_zones:
            if zone.center.z == floor:
                circle = plt.Circle((zone.center.x, zone.center.y), 
                                   zone.radius, 
                                   fill=True, alpha=0.2, color='red',
                                   label=f'干扰：{zone.source_type.value}')
                ax.add_patch(circle)
        
        # 绘制电缆路径
        for path in self.building.cable_paths:
            if any(p.z == floor for p in path.route):
                x_coords = [p.x for p in path.route]
                y_coords = [p.y for p in path.route]
                ax.plot(x_coords, y_coords, '-', 
                       color=self.cable_colors[path.cable_type],
                       linewidth=2, alpha=0.6)
        
        ax.set_xlim(-5, self.building.floor_width + 5)
        ax.set_ylim(-5, self.building.floor_depth + 5)
        ax.set_aspect('equal')
        ax.set_title(f'{self.building.name} - {floor}层平面布线图', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存楼层平面图：{save_path}")
        plt.close()
        
    def _get_room_color(self, room_type: RoomType) -> str:
        """获取房间类型颜色"""
        colors = {
            RoomType.OFFICE: '#87CEEB',
            RoomType.CONFERENCE: '#DDA0DD',
            RoomType.SERVER_ROOM: '#708090',
            RoomType.LOBBY: '#F0E68C',
            RoomType.RESTROOM: '#FFB6C1',
            RoomType.STAIRWELL: '#D3D3D3',
            RoomType.ELEVATOR: '#A9A9A9',
            RoomType.STORAGE: '#DEB887',
        }
        return colors.get(room_type, '#FFFFFF')
    
    def plot_system_architecture(self, save_path: Optional[str] = None):
        """绘制系统架构图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 电缆类型分布
        cable_counts = defaultdict(int)
        cable_lengths = defaultdict(float)
        for path in self.building.cable_paths:
            cable_counts[path.cable_type.value] += 1
            cable_lengths[path.cable_type.value] += path.length
        
        ax1 = axes[0, 0]
        ax1.bar(cable_counts.keys(), cable_counts.values(), color='steelblue')
        ax1.set_title('电缆类型数量分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('电缆类型')
        ax1.set_ylabel('数量')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. 成本分析
        ax2 = axes[0, 1]
        categories = ['水平布线', '垂直干线']
        costs = [sum(p.cost for p in self.building.cable_paths 
                    if p.from_node.startswith('device')),
                sum(p.cost for p in self.building.cable_paths 
                    if p.from_node.startswith('telecom'))]
        ax2.bar(categories, costs, color=['forestgreen', 'darkorange'])
        ax2.set_title('成本分析', fontsize=12, fontweight='bold')
        ax2.set_ylabel('成本 (元)')
        for i, v in enumerate(costs):
            ax2.text(i, v + 100, f'¥{v:.0f}', ha='center', va='bottom')
        
        # 3. 干扰评分分布
        ax3 = axes[1, 0]
        interference_scores = [p.interference_score for p in self.building.cable_paths]
        ax3.hist(interference_scores, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax3.set_title('路径干扰评分分布', fontsize=12, fontweight='bold')
        ax3.set_xlabel('干扰评分')
        ax3.set_ylabel('频数')
        ax3.axvline(np.mean(interference_scores), color='red', 
                   linestyle='--', label=f'平均：{np.mean(interference_scores):.3f}')
        ax3.legend()
        
        # 4. 楼层设备分布
        ax4 = axes[1, 1]
        floor_devices = defaultdict(int)
        for device in self.building.devices.values():
            floor_devices[device.position.z] += 1
        ax4.bar(floor_devices.keys(), floor_devices.values(), color='mediumvioletred')
        ax4.set_title('各楼层设备数量', fontsize=12, fontweight='bold')
        ax4.set_xlabel('楼层')
        ax4.set_ylabel('设备数')
        
        plt.suptitle(f'{self.building.name} - 综合布线系统分析', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存系统分析图：{save_path}")
        plt.close()
    
    def plot_3d_view(self, save_path: Optional[str] = None):
        """绘制 3D 视图"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制楼层平面
        for floor in range(1, self.building.floors + 1):
            z_height = (floor - 1) * 3.5
            ax.plot([0, self.building.floor_width], [0, 0], [z_height, z_height], 
                   'k-', linewidth=1, alpha=0.3)
            ax.plot([0, self.building.floor_width], 
                   [self.building.floor_depth, self.building.floor_depth], 
                   [z_height, z_height], 'k-', linewidth=1, alpha=0.3)
            ax.plot([0, 0], [0, self.building.floor_depth], [z_height, z_height], 
                   'k-', linewidth=1, alpha=0.3)
            ax.plot([self.building.floor_width, self.building.floor_width], 
                   [0, self.building.floor_depth], [z_height, z_height], 
                   'k-', linewidth=1, alpha=0.3)
        
        # 绘制设备
        for device in self.building.devices.values():
            ax.scatter(device.position.x, device.position.y, 
                      (device.position.z - 1) * 3.5,
                      c=self.cable_colors[device.cable_type],
                      s=50, label=device.device_type)
        
        # 绘制电信间
        for tr in self.building.telecom_rooms.values():
            ax.scatter(tr.position.x, tr.position.y, 
                      (tr.position.z - 1) * 3.5,
                      c='blue', s=100, marker='s', label='电信间')
        
        # 绘制电缆路径
        for path in self.building.cable_paths:
            x_coords = [p.x for p in path.route]
            y_coords = [p.y for p in path.route]
            z_coords = [(p.z - 1) * 3.5 for p in path.route]
            ax.plot(x_coords, y_coords, z_coords, 
                   color=self.cable_colors[path.cable_type],
                   linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_zlabel('高度 (米)')
        ax.set_title(f'{self.building.name} - 3D 布线视图', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存 3D 视图：{save_path}")
        plt.close()


# ============================================================================
# 示例：创建智能建筑
# ============================================================================

def create_example_building() -> SmartBuilding:
    """创建示例智能建筑"""
    building = SmartBuilding(
        name="智子科技大厦",
        floors=5,
        floor_area=(40.0, 30.0)  # 40m x 30m
    )
    
    # 添加房间 (每层简化为几个代表性房间)
    for floor in range(1, 6):
        # 办公室
        for i in range(4):
            room = Room(
                id=f"office_{floor}_{i}",
                room_type=RoomType.OFFICE,
                position=Position3D(x=5 + i*8, y=5, z=floor),
                area=24.0,
                work_areas=2,
                bandwidth_requirement=1.0
            )
            building.add_room(room)
        
        # 会议室
        conference = Room(
            id=f"conference_{floor}",
            room_type=RoomType.CONFERENCE,
            position=Position3D(x=35, y=15, z=floor),
            area=36.0,
            work_areas=4,
            bandwidth_requirement=2.0
        )
        building.add_room(conference)
    
    # 添加电信间 (每层一个)
    for floor in range(1, 6):
        tr = TelecommunicationsRoom(
            id=f"telecom_{floor}",
            floor=floor,
            position=Position3D(x=20, y=25, z=floor),
            serves_floors=[floor]
        )
        building.add_telecom_room(tr)
    
    # 设置设备间 (1 层)
    equipment = EquipmentRoom(
        id="equipment_main",
        position=Position3D(x=20, y=25, z=1),
        telecom_rooms=[f"telecom_{f}" for f in range(1, 6)]
    )
    building.set_equipment_room(equipment)
    
    # 添加设备
    device_id = 0
    for floor in range(1, 6):
        # 每层添加 AP
        for i in range(3):
            device = Device(
                id=f"device_{device_id}",
                name=f"AP_{floor}_{i}",
                device_type="AP",
                position=Position3D(x=10 + i*15, y=15, z=floor),
                room_id=f"office_{floor}_{i}",
                bandwidth=1000,  # 1Gbps
                cable_type=CableCategory.CAT6A,
                power_over_ethernet=True
            )
            building.add_device(device)
            device_id += 1
        
        # 每层添加摄像头
        for i in range(2):
            device = Device(
                id=f"device_{device_id}",
                name=f"Camera_{floor}_{i}",
                device_type="Camera",
                position=Position3D(x=5 + i*30, y=28, z=floor),
                room_id=f"office_{floor}_{i}",
                bandwidth=50,  # 50Mbps
                cable_type=CableCategory.CAT6,
                power_over_ethernet=True
            )
            building.add_device(device)
            device_id += 1
    
    # 添加干扰区域
    # 电梯井
    building.add_interference_zone(InterferenceZone(
        id="elevator_shaft",
        source_type=InterferenceSource.ELEVATOR,
        center=Position3D(x=38, y=5, z=1),
        radius=3.0,
        severity=0.8
    ))
    
    # 配电室
    building.add_interference_zone(InterferenceZone(
        id="power_room",
        source_type=InterferenceSource.POWER_LINE,
        center=Position3D(x=2, y=2, z=1),
        radius=5.0,
        severity=0.9
    ))
    
    # 空调机房
    building.add_interference_zone(InterferenceZone(
        id="hvac_room",
        source_type=InterferenceSource.HVAC,
        center=Position3D(x=38, y=28, z=5),
        radius=4.0,
        severity=0.5
    ))
    
    return building


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 70)
    print("智能建筑综合布线系统优化")
    print("Smart Building Integrated Cabling System Optimization")
    print("=" * 70)
    
    # 创建建筑模型
    print("\n[1/4] 创建智能建筑模型...")
    building = create_example_building()
    print(f"✓ 建筑名称：{building.name}")
    print(f"✓ 楼层数：{building.floors}")
    print(f"✓ 楼层面积：{building.floor_width}m × {building.floor_depth}m")
    print(f"✓ 房间数：{len(building.rooms)}")
    print(f"✓ 设备数：{len(building.devices)}")
    print(f"✓ 电信间数：{len(building.telecom_rooms)}")
    print(f"✓ 干扰区域数：{len(building.interference_zones)}")
    
    # 优化路由
    print("\n[2/4] 优化电缆路由...")
    optimizer = CableRoutingOptimizer(building)
    stats = optimizer.optimize_all_routes()
    
    print(f"\n📊 优化结果统计:")
    print(f"  总电缆长度：{stats['total_cable_length']:.2f} 米")
    print(f"  总成本：¥{stats['total_cost']:.2f} 元")
    print(f"  水平电缆数：{stats['horizontal_cables']} 条")
    print(f"  垂直干线数：{stats['backbone_cables']} 条")
    print(f"  平均干扰评分：{stats['avg_interference']:.4f}")
    
    # 可视化
    print("\n[3/4] 生成可视化图表...")
    visualizer = SmartBuildingVisualizer(building, optimizer)
    
    # 绘制典型楼层
    visualizer.plot_floor_plan(floor=3, save_path="outputs/34_smart_building_floor3.png")
    
    # 系统分析
    visualizer.plot_system_architecture(save_path="outputs/34_smart_building_analysis.png")
    
    # 3D 视图
    visualizer.plot_3d_view(save_path="outputs/34_smart_building_3d.png")
    
    # 保存结果
    print("\n[4/4] 保存测试结果...")
    import json
    results = {
        'building_name': building.name,
        'floors': building.floors,
        'total_rooms': len(building.rooms),
        'total_devices': len(building.devices),
        'total_telecom_rooms': len(building.telecom_rooms),
        'statistics': stats,
        'cable_paths_count': len(building.cable_paths),
    }
    
    with open("outputs/34_smart_building_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✓ 保存测试结果：outputs/34_smart_building_results.json")
    
    print("\n" + "=" * 70)
    print("✅ 智能建筑综合布线优化完成!")
    print("=" * 70)
    
    return building, optimizer, stats


if __name__ == "__main__":
    building, optimizer, stats = main()
