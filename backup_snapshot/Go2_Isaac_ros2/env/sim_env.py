import omni
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains import TerrainGeneratorCfg
from env.terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg
from pxr import Gf, UsdGeom, UsdPhysics


NAV_DEMO_OBSTACLES = [
    # x, y, width, depth, height
    (2.2, -0.7, 0.70, 0.70, 0.55),
    (3.4, 0.9, 0.70, 0.70, 0.85),
    (5.1, -1.6, 0.70, 0.70, 0.65),
    (-1.8, 1.2, 0.70, 0.70, 0.50),
]

def add_semantic_label():
    try:
        import omni
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        if not ext_manager.is_extension_enabled("omni.replicator.core"):
            return
        import omni.replicator.core as rep

        ground_plane = rep.get.prims("/World/GroundPlane")
        with ground_plane:
        # Add a semantic label
            rep.modify.semantics([("class", "floor")])
    except Exception as exc:
        print(f"[WARN] Skipping semantic label setup: {exc}", flush=True)

def create_obstacle_env():
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=100 ,                   #use 200,400 to add more obstacles or 0 for plane env--------------------
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 


def create_nav_demo_env():
    stage = omni.usd.get_context().get_stage()
    root = define_prim("/World/NavDemoObstacles", "Xform")
    for idx, (x, y, width, depth, height) in enumerate(NAV_DEMO_OBSTACLES):
        prim_path = f"/World/NavDemoObstacles/box_{idx}"
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, y, height * 0.5))
        cube.AddScaleOp().Set(Gf.Vec3f(width, depth, height))
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)


def create_warehouse_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse.usd"     #also--> full_warehouse.usd,, warehouse_with_forklifts.usd ,, warehouse_multiple_shelves.usd
    prim.GetReferences().AddReference(asset_path)

def create_hospital_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Hospital")
    prim = define_prim("/World/Hospital", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Hospital/hospital.usd"
    prim.GetReferences().AddReference(asset_path)

def create_office_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Office")
    prim = define_prim("/World/Office", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Office/office.usd"
    prim.GetReferences().AddReference(asset_path)

def create_rivermark_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Outdoor")
    prim = define_prim("/World/Outdoor", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Outdoor/Rivermark/rivermark.usd"
    prim.GetReferences().AddReference(asset_path)

def create_terrain_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Terrains")
    prim = define_prim("/World/Terrains", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Terrains/slope.usd"      #also--> rough_plane.usd,, stairs.usd,, slope.usd
    prim.GetReferences().AddReference(asset_path)
