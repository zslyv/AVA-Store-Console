import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.sql import text
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import heapq

usuario = 'root'
password = '-Ime44SqlD0lph'
host = 'localhost'
puerto = 3306
base_de_datos = 'store'
cadena_conexion = f"mysql+pymysql://{usuario}:{password}@{host}:{puerto}/{base_de_datos}"
engine = create_engine(cadena_conexion)

consulta_productos = "SELECT * FROM productos"
consulta_compras = "SELECT * FROM compras"
df_productos = pd.read_sql(consulta_productos, engine)
df_compras = pd.read_sql(consulta_compras, engine)

df_compras['compra_id'] = df_compras['fecha'].astype(str) + ' ' + df_compras['hora'].astype(str)

matriz_usuario_producto = df_compras.pivot_table(index='compra_id', columns='producto_id', values='cantidad', fill_value=0)

matriz_co_compras = matriz_usuario_producto.T.dot(matriz_usuario_producto)

popularidad_productos = df_compras.groupby('producto_id').size().to_dict()

scaler = StandardScaler()
matriz_usuario_producto_scaled = scaler.fit_transform(matriz_usuario_producto)

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(matriz_usuario_producto_scaled)

class Graph:
    def __init__(self):
        self.edges = {}
        
    def neighbors(self, node):
        return self.edges.get(node, [])
    
    def add_edge(self, from_node, to_node, weight):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append((to_node, weight))

def heuristic(node, goal):
    return 0

def a_star_search(graph, start, goal):
    pq = []
    heapq.heappush(pq, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while pq:
        current_priority, current_node = heapq.heappop(pq)
        
        if current_node == goal:
            break
        
        for neighbor, weight in graph.neighbors(current_node):
            new_cost = cost_so_far[current_node] + weight
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(pq, (priority, neighbor))
                came_from[neighbor] = current_node
    
    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def construir_grafo(df_compras):
    graph = Graph()
    co_compras = df_compras.pivot_table(index='usuario_id', columns='producto_id', values='cantidad', fill_value=0)
    co_compra_matrix = co_compras.T.dot(co_compras)
    
    for producto_id in co_compra_matrix.columns:
        for co_producto_id in co_compra_matrix.index:
            if producto_id != co_producto_id:
                graph.add_edge(producto_id, co_producto_id, 1 / (co_compra_matrix[producto_id][co_producto_id] + 1))
    
    return graph

graph = construir_grafo(df_compras)

def obtener_recomendaciones(productos_seleccionados, n_recomendaciones=5):
    recomendaciones = {}

    for producto_id in productos_seleccionados:
        if producto_id in matriz_co_compras.columns:
            co_compras = matriz_co_compras[producto_id]
            for co_producto_id, cantidad in co_compras.items():
                if co_producto_id not in productos_seleccionados:
                    if co_producto_id not in recomendaciones:
                        recomendaciones[co_producto_id] = cantidad
                    else:
                        recomendaciones[co_producto_id] += cantidad

    usuario_input = np.zeros(matriz_usuario_producto.shape[1])
    for producto_id in productos_seleccionados:
        if producto_id - 1 < len(usuario_input):
            usuario_input[producto_id - 1] = 1
    
    usuario_input_scaled = scaler.transform([usuario_input])
    distancias, indices = knn.kneighbors(usuario_input_scaled, n_neighbors=5)
    
    for indice in indices.flatten():
        similar_items = matriz_usuario_producto.iloc[indice]
        for producto_id, cantidad in similar_items.items():
            if producto_id not in productos_seleccionados and cantidad > 0:
                if producto_id not in recomendaciones:
                    recomendaciones[producto_id] = cantidad
                else:
                    recomendaciones[producto_id] += cantidad

    recomendaciones = {k: v for k, v in recomendaciones.items() if k not in productos_seleccionados}

    top_recomendaciones_astar = []
    for producto in productos_seleccionados:
        try:
            path = a_star_search(graph, producto, productos_seleccionados[-1])
            top_recomendaciones_astar.extend(path[1:])
        except KeyError as e:
            print(f"Error en A*: {e}")

    for producto_id in top_recomendaciones_astar:
        if producto_id not in productos_seleccionados and producto_id not in recomendaciones:
            recomendaciones[producto_id] = 1

    recomendaciones_ordenadas = sorted(recomendaciones.items(), key=lambda x: (recomendaciones[x[0]], popularidad_productos.get(x[0], 0)), reverse=True)
    top_recomendaciones = [producto_id for producto_id, cantidad in recomendaciones_ordenadas[:n_recomendaciones]]
    
    return top_recomendaciones

productos_seleccionados = [1, 2, 3, 4]
recomendaciones = obtener_recomendaciones(productos_seleccionados, n_recomendaciones=5)
productos_recomendados = df_productos[df_productos['id'].isin(recomendaciones)]

nombres_productos_seleccionados = df_productos[df_productos['id'].isin(productos_seleccionados)]['nombre'].tolist()

print("-----------------------------------------")
print("Recomendaciones basadas en los productos seleccionados:", productos_seleccionados)
print("Productos seleccionados:")
print("")
for nombre in nombres_productos_seleccionados:
    print(nombre)
print("")
print("-----------------------------------------")
print("Productos recomendados:")
print(productos_recomendados[['nombre', 'categoria']])

usuario_id = 1

try:
    with engine.connect() as conexion:
        with conexion.begin() as transaccion:
            for producto_id in productos_seleccionados:
                consulta_insertar_compras = text("INSERT INTO compras (usuario_id, producto_id, cantidad, fecha, hora) VALUES (:usuario_id, :producto_id, 1, CURDATE(), CURTIME())")
                conexion.execute(consulta_insertar_compras, {'usuario_id': usuario_id, 'producto_id': producto_id})

            for producto_id in recomendaciones:
                consulta_insertar_recomendaciones = text("INSERT INTO recomendaciones (usuario_id, producto_id, fecha) VALUES (:usuario_id, :producto_id, CURDATE())")
                conexion.execute(consulta_insertar_recomendaciones, {'usuario_id': usuario_id, 'producto_id': producto_id})

except exc.SQLAlchemyError as e:
    print(f"Error al insertar en la base de datos: {e}")
