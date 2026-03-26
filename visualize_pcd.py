import open3d as o3d

pcd_object = o3d.io.read_point_cloud("3d_construction/outputs/time_fuse_chair_custom/mask_track_000.pcd")
pcd_office = o3d.io.read_point_cloud("3d_construction/outputs/fr3_office_masked_object.pcd")
# pcd = o3d.io.read_point_cloud("drive-download-20260324T090340Z-3-001/000000_pred_nms_bbox.ply")
o3d.visualization.draw_geometries([pcd_office])

o3d.visualization.draw_geometries([pcd_object])