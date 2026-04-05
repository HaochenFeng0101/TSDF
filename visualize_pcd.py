import open3d as o3d

pcd_object = o3d.io.read_point_cloud("3d_construction/outputs/time_fuse_chair_custom/mask_track_002.pcd")

pcd_office = o3d.io.read_point_cloud("3d_construction/outputs/fuse_obj_fr3_office_chair_yolov8x-seg_20260403_154353_20260403_160750/mask_track_000.pcd")

# pcd = o3d.io.read_point_cloud("drive-download-20260324T090340Z-3-001/000000_pred_nms_bbox.ply")
o3d.visualization.draw_geometries([pcd_office])


o3d.visualization.draw_geometries([pcd_object])