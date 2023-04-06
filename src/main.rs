#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unused_must_use)]
#![deny(unused_mut)]

//! This crate contains all of the code for the sd-bulk-webui server.

mod metaprompt;

use anyhow::{bail, Context, Error, Result};
use async_std::{fs, io::WriteExt, path::PathBuf, stream::StreamExt};
use axum::{
    body::Bytes,
    extract::{Path, Query},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use hyper;
use metaprompt::generate_prompts;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    io::{BufReader, Write},
    net::SocketAddr,
};
use tower_http::services::ServeDir;

// CONTRIBUTE: The way we handle pngs is highly inefficient. From a reliability standpoint, the png
// decoder and encoder that we use is synchronous, which means that large png images are going to
// freeze a thread and that could cause considerable instability in the API if the png being
// processed is too large. The other optimization is that when we receive the png image from stable
// diffusion, we decode the entire image data and then re-encode it. We could just copy the image
// data chunk over instead, which would save us a considerable amount of CPU power.

// CONTRIBUTE: This is a larger amount of work, but it would be great if someone built out some IDE
// features for the metaprompt text editor. Highlight syntax errors and unused nodes, highlight
// words that lead to 'bad' or 'fantastic' prompts more frequently, etc.

// TODO: Need to make sure we are setting the model correctly
//
// TODO: Need to make sure we aren't running two batches at the same time

// TODO: Need to update so that we change the model when switching between metaprompts.
//
// TODO: Need to switch generate_imgs to websockets so that the ui can get progress updates.

// Establish the routes for the bulkui endpoints.
fn routes() -> Router {
    // Build the router with each route.
    Router::new()
        .nest_service("/", ServeDir::new("html"))
        .nest_service("/app_data", ServeDir::new("app_data"))
        .route("/gallery/:name", get(gallery))
        .route("/generate_imgs/:name", post(generate_imgs))
        .route("/metaprompt/:name", get(metaprompt).post(metaprompt_save))
        .route("/metaprompt_list", get(metaprompt_list))
        // .route("/save", post(save))
        .route("/update_tags", post(update_tags))
}

// Spin up the server that manages sd-bulk-webui.
#[tokio::main]
async fn main() {
    let addr = SocketAddr::from(([127, 0, 0, 1], 7865));
    let app = routes();

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

// err_bad_request is a helper method for returning a bad request error.
fn err_bad_request(e: Error) -> (StatusCode, Response) {
    (StatusCode::BAD_REQUEST, format!("{:#}", e).into_response())
}

// err_404 is a helper method for returning a not found error.
fn err_404(e: Error) -> (StatusCode, Response) {
    (StatusCode::NOT_FOUND, format!("{:#}", e).into_response())
}

// err_internal is a helper method for returning internal server errors.
fn err_internal(e: Error) -> (StatusCode, Response) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("{:#}", e).into_response(),
    )
}

// returns true if the provided string is a valid metaprompt name.
fn valid_metaprompt_name(s: &str) -> bool {
    // Check for non-ascii characters.
    if !s.is_ascii() {
        return false;
    }
    let b = s.as_bytes();

    // Check for bad characters.
    for i in 0..b.len() {
        if b[i] < 32 {
            return false;
        }
        if b[i] == 127 {
            return false;
        }
        if b[i] == 47 {
            return false;
        }
        if b[i] == 92 {
            return false;
        }
    }
    true
}

// GalleryImgData defines the data that gets connected to a single image in a metaprompt gallery.
#[derive(Serialize)]
struct GalleryImgData {
    location: String,
    batch: u64,
    tags: Vec<String>,
    prompt_elements: Vec<String>,
}

// gallery_handler performs all of the processing for the gallery endpoint.
//
// The return value is a vector of tuples, where each tuple contains the location of an image, the
// batch number of the image, and the tags of the image.
async fn gallery_handler(name: String) -> Result<Vec<GalleryImgData>, Error> {
    let mut imgs_data: Vec<GalleryImgData> = Vec::new();

    // Get the base data path for this metaprompt.
    let mut data_path = PathBuf::new();
    data_path.push("app_data");
    data_path.push(name.clone());
    if !data_path.is_dir().await {
        bail!("unable to find metaprompt data");
    }

    // Open each batch directory and parse through all the images.
    let mut entries = fs::read_dir(&data_path)
        .await
        .context("unable to read metaprompt dir")?;
    while let Some(res) = entries.next().await {
        let entry = match res {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip if this is not a directory
        let path = entry.path();
        if !path.is_dir().await {
            continue;
        }
        let dir_str = match path.file_name() {
            Some(d) => d,
            None => continue,
        }
        .to_string_lossy()
        .to_string();

        // Ignore the directory if it doesn't have a valid batch name.
        if !dir_str.starts_with("batch_") {
            continue;
        }
        let batch_num_str = match dir_str.strip_prefix("batch_") {
            Some(s) => s,
            None => continue,
        };
        let batch_num = match batch_num_str.parse::<u64>() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Grab every image in the directory.
        let mut entries = match fs::read_dir(&path).await {
            Ok(n) => n,
            Err(_) => continue,
        };
        while let Some(res) = entries.next().await {
            let entry = match res {
                Ok(e) => e,
                Err(_) => continue,
            };

            // Skip if it's not a png file.
            let path = entry.path();
            if !path.is_file().await {
                continue;
            }
            if !path.to_string_lossy().to_string().ends_with(".png") {
                continue;
            }

            // Open the file and parse the metadata.
            let file = match std::fs::File::open(path.clone()) {
                Ok(f) => f,
                Err(_) => continue,
            };
            tokio::task::yield_now().await;
            let decoder = png::Decoder::new(file);
            tokio::task::yield_now().await;
            let info_reader = match decoder.read_info() {
                Ok(n) => n,
                Err(_) => continue,
            };
            tokio::task::yield_now().await;
            let mut img_data = GalleryImgData {
                location: path.to_string_lossy().to_string(),
                batch: batch_num,
                tags: Vec::new(),
                prompt_elements: Vec::new(),
            };
            for text_chunk in &info_reader.info().uncompressed_latin1_text {
                tokio::task::yield_now().await;
                if text_chunk.keyword == "tags" {
                    let text = text_chunk.text.trim_end_matches('.');
                    let tags = text.split(", ").map(|x| x.to_string()).collect();
                    img_data.tags = tags;
                }
                if text_chunk.keyword == "bulkui_prompt_args" {
                    let json_value: Value = match serde_json::from_str(&text_chunk.text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    match json_value.get("prompt") {
                        Some(vp) => {
                            if let Value::String(p) = vp {
                                let elements: Vec<&str> = p.split(", ").collect();
                                for elem in elements {
                                    img_data.prompt_elements.push(elem.to_string());
                                }
                            }
                        }
                        None => {}
                    };
                    match json_value.get("negative_prompt") {
                        Some(vn) => {
                            if let Value::String(n) = vn {
                                let elements: Vec<&str> = n.split(", ").collect();
                                for elem in elements {
                                    img_data.prompt_elements.push(format!("!{}", elem));
                                }
                            }
                        }
                        None => {}
                    };
                }
            }
            imgs_data.push(img_data);
        }
    }
    Ok(imgs_data)
}

// gallery returns all of the images for a provided metaprompt.
async fn gallery(Path(name): Path<String>) -> impl IntoResponse {
    match gallery_handler(name).await {
        Ok(i) => (StatusCode::OK, Json(i).into_response()),
        Err(e) => err_404(e),
    }
}

// The handler logic for metaprompt_list
async fn metaprompt_list_handler() -> Result<Vec<String>, Error> {
    let mut metaprompts: Vec<String> = Vec::new();
    let mut entries = fs::read_dir("./app_data")
        .await
        .context("unable to find app_data directory")?;
    while let Some(res) = entries.next().await {
        let entry = res.context("unable to fully process app_data directory")?;

        // Skip if this is not a directory
        let mut path = entry.path();
        if !path.is_dir().await {
            continue;
        }

        // Skip any directories with invalid characters.
        let path_str = path
            .file_name()
            .context("unable to get filename from file")?
            .to_string_lossy()
            .to_string();

        if !valid_metaprompt_name(&path_str) {
            continue;
        }

        // Look for a metaprompt in the directory.
        path.push("metaprompt.mp");
        if !path.is_file().await {
            continue;
        }
        metaprompts.push(path_str);
    }
    Ok(metaprompts)
}

// Establish a handler to load all of the metaprompts in the user's directory.
async fn metaprompt_list() -> impl IntoResponse {
    match metaprompt_list_handler().await {
        Ok(metaprompts) => return (StatusCode::OK, Json(metaprompts).into_response()),
        Err(e) => err_internal(e),
    }
}

// Establish a handler to return the metaprompt associate with the given name.
async fn metaprompt(Path(name): Path<String>) -> impl IntoResponse {
    let mut data_path = PathBuf::new();
    data_path.push("app_data");
    data_path.push(name);
    data_path.push("metaprompt.mp");
    match fs::read(data_path).await {
        Ok(data) => (StatusCode::OK, data.into_response()),
        Err(e) => err_404(Error::new(e)),
    }
}

// Establish a handler to save a metaprompt.
async fn metaprompt_save(Path(name): Path<String>, body: Bytes) -> impl IntoResponse {
    let mut data_path = PathBuf::new();
    data_path.push("app_data");
    data_path.push(name);
    data_path.push("metaprompt.mp");
    if !data_path.is_file().await {
        return err_404(Error::msg("cannot update prompt"));
    }
    match fs::write(data_path, body).await {
        Ok(_) => (StatusCode::OK, "success\n".to_string().into_response()),
        Err(e) => err_internal(Error::new(e)),
    }
}

// generate_img will generate one image for generate_imgs.
async fn generate_img(
    metaprompt_name: String,
    batch_number: u64,
    prompt_args: String,
    batch_dir: &PathBuf,
    img_num: u64,
    set_name: String,
    author_name: String,
    checkpoint: String,
    vae: String,
) -> Result<(), Error> {
    // Establish the batch metadata.
    let batch_args = format!(
        r###"{{"metaprompt": "{}", "batch": {}, "set": "{}", "author": "{}", "checkpoint": "{}", "vae": "{}"}}"###,
        metaprompt_name, batch_number, set_name, author_name, checkpoint, vae
    );

    // Use hyper to make a POST request that will return a txt2img result.
    let client = hyper::Client::new();
    let req = hyper::Request::builder()
        .method(hyper::Method::POST)
        .uri("http://localhost:7860/sdapi/v1/txt2img")
        .header("user-agent", "sd-build-webui-server")
        .header("content-type", "application/json")
        .body(hyper::Body::from(prompt_args.clone()))
        .context("unable to make request")?;
    let resp = client
        .request(req)
        .await
        .context("unable to fetch response")?;
    let body_bytes = hyper::body::to_bytes(resp.into_body())
        .await
        .context("unable to parse response body")?;
    let body_str = std::str::from_utf8(body_bytes.as_ref())
        .context("unable to parse response body to string")?;

    // Decode the image from the json.
    let json_obj: Value =
        serde_json::from_str(body_str).context("unable to parse response json")?;
    let images_obj = match json_obj.get("images") {
        Some(o) => o,
        None => bail!("{}", body_str),
    };
    let images = images_obj
        .as_array()
        .context("unable to parse images array")?;
    if images.len() < 1 {
        panic!("expecting at least one image");
    }
    for i in 0..images.len() {
        let img_b64 = images[i]
            .as_str()
            .context("unable to parse string from images array")?;
        let img_bytes = base64::engine::general_purpose::STANDARD
            .decode(img_b64)
            .context("unable to read base64 image data")?;

        // Decode the parameters from the json.
        let param_str = json_obj
            .get("parameters")
            .context("unable to read image parameters")?
            .to_string();

        // Pull out the png data.
        let img_reader = BufReader::new(&img_bytes[..]);
        let decoder = png::Decoder::new(img_reader);
        let mut reader = decoder.read_info().context("unable to get png metadata")?;
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buf)
            .context("unable to read first frame of image data")?;
        let bytes = &buf[..info.buffer_size()];

        // Establish the default set of tags for the image.
        let tags = format!(
            "ai, stable diffusion, {}, {}, {}, {}",
            set_name, author_name, checkpoint, vae
        );
        let filler = ".".repeat(2048 - tags.len());
        let tags = tags + &filler;

        // Save the file in the proper folder as a png, where the png has a large area allocated for
        // tags. Add as metadata the input we used, the prompt recorded by webui, and leave space for
        // user tags.
        let mut buf = Vec::new();
        let mut encoder = png::Encoder::new(&mut buf, info.width, info.height);
        tokio::task::yield_now().await;
        encoder.set_color(info.color_type);
        encoder.set_depth(info.bit_depth);
        encoder.set_compression(png::Compression::Best);
        encoder
            .add_text_chunk("webui_parameters".to_string(), param_str)
            .context("unable to add webui_parameters text chunk")?;
        encoder
            .add_text_chunk("bulkui_prompt_args".to_string(), prompt_args.clone())
            .context("unable to add bulkui_prompt_args text chunk")?;
        encoder
            .add_text_chunk("bulkui_batch_data".to_string(), batch_args.clone())
            .context("unable to add bulkui_batch_data text chunk")?;
        encoder
            .add_text_chunk("tags".to_string(), tags)
            .context("unable to add tags text chunk")?;
        tokio::task::yield_now().await;
        let mut writer = encoder
            .write_header()
            .context("unable to write png header")?;
        tokio::task::yield_now().await;
        writer
            .write_image_data(&bytes)
            .context("unable to write png image data")?;
        tokio::task::yield_now().await;
        drop(writer);
        tokio::task::yield_now().await;

        let mut img_path = batch_dir.clone();
        img_path.push(format!("{}.{}.png", img_num, i));
        let mut outfile = fs::File::create(img_path)
            .await
            .context("unable to open image file for writing")?;
        outfile
            .write_all(&buf)
            .await
            .context("unable to write image data to disk")?;
    }
    Ok(())
}

// Contains all the recognized query params for the generate_imgs route.
#[derive(Deserialize)]
struct GenerateImgsQueryParams {
    batch_size: u64,
}

// Establish a handler for the 'generate' button.
async fn generate_imgs(
    Path(name): Path<String>,
    query_params: Query<GenerateImgsQueryParams>,
    body: Bytes,
) -> impl IntoResponse {
    // Get the base data path for this metaprompt.
    let mut data_path = PathBuf::new();
    data_path.push("app_data");
    data_path.push(name.clone());

    // Return an error if the metaprompt dir doesn't exist.
    if !data_path.is_dir().await {
        return err_404(Error::msg("metaprompt not found"));
    }

    // Scroll through the directories and identify the largest batch number.
    let mut latest_batch = 1;
    let mut entries = match fs::read_dir(&data_path).await {
        Ok(entries) => entries,
        Err(e) => return err_internal(Error::new(e)),
    };
    while let Some(res) = entries.next().await {
        let entry = match res {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip if this is not a directory
        let path = entry.path();
        if !path.is_dir().await {
            continue;
        }

        // Skip any directories with invalid characters.
        let file_name = match path.file_name() {
            Some(n) => n,
            None => continue,
        };
        let path_str = file_name.to_string_lossy().to_string();

        // Ignore the directory if it doesn't have a batch prefix.
        if !path_str.starts_with("batch_") {
            continue;
        }
        let batch_num_str = match path_str.strip_prefix("batch_") {
            Some(s) => s,
            None => continue,
        };
        match batch_num_str.parse::<u64>() {
            Ok(n) => {
                if n > latest_batch && query_params.batch_size > 200 {
                    latest_batch = n;
                }
            }
            Err(_) => continue,
        };
    }

    // Check whether the metaprompt in the latest batch matches the metaprompt we are using for
    // this batch. If the dir or metaprompt file doesn't exist, we will keep the batch number. If
    // the metaprompt file does exist and it doesn't match the metaprompt we have, we will increase
    // the batch number.
    let latest_batch_dirname = format!("batch_{}", latest_batch);
    let mut metaprompt_filename = data_path.clone();
    metaprompt_filename.push(latest_batch_dirname);
    metaprompt_filename.push("metaprompt.mp");
    if metaprompt_filename.is_file().await {
        match fs::read(metaprompt_filename).await {
            Ok(metaprompt_bytes) => {
                if body.as_ref() != metaprompt_bytes {
                    latest_batch += 1;
                }
            }
            Err(_) => {}
        }
    }
    let latest_batch_dirname = format!("batch_{}", latest_batch);

    // Make sure the batch directory exists.
    data_path.push(latest_batch_dirname);
    match fs::create_dir_all(&data_path).await {
        Ok(()) => {}
        Err(e) => return err_internal(Error::new(e)),
    }

    // Save the metaprompt with the batch.
    let mut metaprompt_path = data_path.clone();
    metaprompt_path.push("metaprompt.mp");
    match fs::write(metaprompt_path, &body).await {
        Ok(_) => {}
        Err(e) => return err_internal(Error::new(e)),
    };

    // Determine the largest image index in the batch dir.
    let mut largest_img_num = 0;
    let mut entries = match fs::read_dir(&data_path).await {
        Ok(entries) => entries,
        Err(e) => return err_internal(Error::new(e)),
    };
    while let Some(res) = entries.next().await {
        let entry = match res {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip if this is not a file
        let path = entry.path();
        if !path.is_file().await {
            continue;
        }

        // Skip any files that are not *.png.
        let file_name = match path.file_name() {
            Some(n) => n,
            None => continue,
        };
        let path_str = file_name.to_string_lossy().to_string();
        if !path_str.ends_with(".png") {
            continue;
        }
        // Determine the number for this image.
        let img_num = path_str
            .chars()
            .take_while(|c| c.is_numeric())
            .collect::<String>();
        match img_num.parse::<u64>() {
            Ok(n) => {
                if n > largest_img_num {
                    largest_img_num = n;
                }
            }
            Err(_) => continue,
        };
    }

    // Generate a list of prompts from the metaprompt.
    let gen_result = match generate_prompts(&body, query_params.batch_size) {
        Ok(r) => r,
        Err(e) => return err_bad_request(e),
    };
    let prompts = gen_result.prompts;

    for i in 0..prompts.len() {
        match generate_img(
            name.clone(),
            latest_batch,
            prompts[i].clone(),
            &data_path,
            largest_img_num + 1 + i as u64,
            gen_result.set_name.clone(),
            gen_result.set_author.clone(),
            gen_result.checkpoint.clone(),
            gen_result.vae.clone(),
        )
        .await
        {
            Ok(_) => {} // TODO: Some sort of websocket update so the app can follow along as we make progress
            Err(e) => return err_internal(e),
        }
    }
    (StatusCode::OK, "success\n".to_string().into_response())
}

// Implement the business logic for the update_tags endpoint. If possible, we only update the
// header. If necessary, we will re-write the whole image. The whole image only needs a rewrite if
// there is no tags header, or if the tags header is too small.
async fn update_tags_handler(img_path: PathBuf, tags_str: String) -> Result<(), Error> {
    let update_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(img_path.clone())
        .context("unable to open image file")?;
    tokio::task::yield_now().await;
    let decoder = png::Decoder::new(update_file);
    tokio::task::yield_now().await;
    let mut info_reader = decoder.read_info().context("unable to read image header")?;
    tokio::task::yield_now().await;
    let info = info_reader.info();
    tokio::task::yield_now().await;

    // As we read through the header info of the previous image, start writing out the updated
    // image data.
    let mut buf = Vec::new();
    let mut encoder = png::Encoder::new(&mut buf, info.width, info.height);
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);
    encoder.set_compression(png::Compression::Best);
    let mut tags_found = false;
    tokio::task::yield_now().await;
    for text_chunk in &info.uncompressed_latin1_text {
        tokio::task::yield_now().await;
        if text_chunk.keyword != "tags" {
            encoder
                .add_text_chunk(text_chunk.keyword.clone(), text_chunk.text.clone())
                .context("unable to add text chunk")?;
            continue;
        }

        // Check that the new tags fit inside of the existing tag list. If not, need to treat the
        // tags header as though it doesn't exist.
        if tags_str.len() > text_chunk.text.len() {
            continue;
        }

        // Replace the existing tags header.
        tags_found = true;
        let filler = ".".repeat(text_chunk.text.len() - tags_str.len());
        let tags_data = tags_str.clone() + &filler;
        encoder
            .add_text_chunk("tags".to_string(), tags_data)
            .context("unable to add tags chunk")?;
    }

    // If we found the tags header, we only need to write the header of this image. Otherwise we
    // need to write out the whole image.
    tokio::task::yield_now().await;
    if tags_found {
        encoder
            .write_header()
            .context("unable to write image header")?;

        // Write the encoded data to the file.
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(img_path)
            .context("unable to open image file for writing")?;
        tokio::task::yield_now().await;
        file.write_all(&buf[..buf.len() - 12])
            .context("unable to write image header data to disk")?;
    } else {
        // Write the tags header.
        let mut filled_len = tags_str.len() * 2;
        if filled_len < 2048 {
            filled_len = 2048;
        }
        let filler = ".".repeat(filled_len - tags_str.len());
        let tags_data = tags_str + &filler;
        encoder
            .add_text_chunk("tags".to_string(), tags_data)
            .context("unable to add fresh tags chunk to image")?;
        tokio::task::yield_now().await;

        // Write out the image data as well.
        let mut img_buf = vec![0; info_reader.output_buffer_size()];
        info_reader
            .next_frame(&mut img_buf)
            .context("unable to read image frame")?;
        tokio::task::yield_now().await;
        let mut writer = encoder
            .write_header()
            .context("unable to write header to buffer")?;
        writer
            .write_image_data(&img_buf)
            .context("unable to write image data to buffer")?;
        tokio::task::yield_now().await;
        drop(writer);

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(img_path)
            .context("unable to open image file for writing")?;
        tokio::task::yield_now().await;
        file.write_all(&buf)
            .context("unable to write full image data to disk")?;
    }
    tokio::task::yield_now().await;
    Ok(())
}

// The json parameters of the request body for update_tags
#[derive(Deserialize)]
struct UpdateTagsPostParams {
    path: String,
    tags: Vec<String>,
}

// the handler for the update_tags endpoint
async fn update_tags(Json(body): Json<UpdateTagsPostParams>) -> impl IntoResponse {
    // Open the image file and find the list of tags. If the list of tags can't be found, we'll
    // rewrite the whole image.
    let mut img_path = PathBuf::new();
    img_path.push(body.path);
    if !img_path.is_file().await {
        return err_404(Error::msg("img not found"));
    }

    // Attempt to update the tags, return an internal server error if it fails.
    let tags_str = body.tags.join(", ");
    match update_tags_handler(img_path, tags_str).await {
        Ok(()) => (StatusCode::OK, "success\n".into_response()),
        Err(e) => err_internal(e),
    }
}

// save implements
async fn save(Json(body): Json<Vec<String>>) -> impl IntoResponse {}
