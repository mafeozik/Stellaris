#include "ppm_image.h"

#include <array>
#include <fstream>
#include <stdio.h>
#include <setjmp.h>

#include <jpeglib.h>

using namespace std;

namespace img_lib {

// структура из примера LibJPEG
struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

typedef struct my_error_mgr* my_error_ptr;

// функция из примера LibJPEG
METHODDEF(void)
my_error_exit (j_common_ptr cinfo) {
    my_error_ptr myerr = (my_error_ptr) cinfo->err;
    (*cinfo->err->output_message) (cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

// В эту функцию вставлен адаптированный код примера write_JPEG_file
// из библиотеки libjpeg. Логика работы (7 шагов) сохранена, но
// убраны глобальные переменные, задание качества, диагностические
// сообщения и аварийный выход через exit().
bool SaveJPEG(const Path& file, const Image& image) {
    /* Структура с параметрами компрессии JPEG.
     * В C++ слово struct при объявлении переменной не требуется.
     */
    jpeg_compress_struct cinfo;
    /* Стандартный обработчик ошибок. */
    jpeg_error_mgr jerr;
    /* Прочее */
    FILE* outfile;            /* целевой файл */
    JSAMPROW row_pointer[1];  /* указатель на строку JSAMPLE */
    int row_stride;           /* физическая ширина строки в буфере */

    /* Шаг 1: выделяем память и инициализируем объект компрессии JPEG */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    /* Шаг 2: задаём приёмник данных (файл).
     * Под Visual Studio для открытия файла используем _wfopen,
     * чтобы корректно работать с путями, содержащими не-ASCII символы.
     */
#ifdef _MSC_VER
    if ((outfile = _wfopen(file.wstring().c_str(), L"wb")) == NULL) {
#else
    if ((outfile = fopen(file.string().c_str(), "wb")) == NULL) {
#endif
        // Раньше здесь были fprintf и exit(1) — теперь просто
        // освобождаем объект компрессии и возвращаем false.
        jpeg_destroy_compress(&cinfo);
        return false;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    /* Шаг 3: задаём параметры компрессии.
     * Размеры берём из параметра image, а не из глобальных переменных.
     */
    cinfo.image_width = image.GetWidth();   /* ширина изображения в пикселях */
    cinfo.image_height = image.GetHeight(); /* высота изображения в пикселях */
    cinfo.input_components = 3;             /* число цветовых компонент в пикселе */
    cinfo.in_color_space = JCS_RGB;         /* цветовое пространство входа */

    /* Устанавливаем параметры компрессии по умолчанию.
     * Вызов jpeg_set_quality убран — используется качество по умолчанию.
     */
    jpeg_set_defaults(&cinfo);

    /* Шаг 4: запускаем компрессор */
    jpeg_start_compress(&cinfo, TRUE);

    /* Шаг 5: пока остаются строки изображения — записываем их */
    row_stride = image.GetWidth() * 3; /* число JSAMPLE в одной строке буфера */

    /* Данные изображения хранятся в виде Color (R,G,B,A), а libjpeg
     * ожидает плотный массив байтов в порядке R,G,B. Поэтому каждую
     * строку перепаковываем в отдельный буфер.
     */
    std::vector<JSAMPLE> buffer(row_stride);

    while (cinfo.next_scanline < cinfo.image_height) {
        const Color* line = image.GetLine(cinfo.next_scanline);
        for (int x = 0; x < image.GetWidth(); ++x) {
            buffer[x * 3 + 0] = std::to_integer<JSAMPLE>(line[x].r);
            buffer[x * 3 + 1] = std::to_integer<JSAMPLE>(line[x].g);
            buffer[x * 3 + 2] = std::to_integer<JSAMPLE>(line[x].b);
        }
        row_pointer[0] = buffer.data();
        (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    /* Шаг 6: завершаем компрессию */
    jpeg_finish_compress(&cinfo);
    /* После finish_compress можно закрыть выходной файл. */
    fclose(outfile);

    /* Шаг 7: освобождаем объект компрессии JPEG */
    jpeg_destroy_compress(&cinfo);

    /* Готово. */
    return true;
}

// тип JSAMPLE фактически псевдоним для unsigned char
void SaveSсanlineToImage(const JSAMPLE* row, int y, Image& out_image) {
    Color* line = out_image.GetLine(y);
    for (int x = 0; x < out_image.GetWidth(); ++x) {
        const JSAMPLE* pixel = row + x * 3;
        line[x] = Color{byte{pixel[0]}, byte{pixel[1]}, byte{pixel[2]}, byte{255}};
    }
}

Image LoadJPEG(const Path& file) {
    jpeg_decompress_struct cinfo;
    my_error_mgr jerr;

    FILE* infile;
    JSAMPARRAY buffer;
    int row_stride;

    // Тут не избежать функции открытия файла из языка C,
    // поэтому приходится использовать конвертацию пути к string.
    // Под Visual Studio это может быть опасно, и нужно применить
    // нестандартную функцию _wfopen
#ifdef _MSC_VER
    if ((infile = _wfopen(file.wstring().c_str(), L"rb")) == NULL) {
#else
    if ((infile = fopen(file.string().c_str(), "rb")) == NULL) {
#endif
        return {};
    }

    /* Шаг 1: выделяем память и инициализируем объект декодирования JPEG */

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return {};
    }

    jpeg_create_decompress(&cinfo);

    /* Шаг 2: устанавливаем источник данных */

    jpeg_stdio_src(&cinfo, infile);

    /* Шаг 3: читаем параметры изображения через jpeg_read_header() */

    (void) jpeg_read_header(&cinfo, TRUE);

    /* Шаг 4: устанавливаем параметры декодирования */

    // установим желаемый формат изображения
    cinfo.out_color_space = JCS_RGB;
    cinfo.output_components = 3;

    /* Шаг 5: начинаем декодирование */

    (void) jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width * cinfo.output_components;

    buffer = (*cinfo.mem->alloc_sarray)
                ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    /* Шаг 5a: выделим изображение ImgLib */
    Image result(cinfo.output_width, cinfo.output_height, Color::Black());

    /* Шаг 6: while (остаются строки изображения) */
    /*                     jpeg_read_scanlines(...); */

    while (cinfo.output_scanline < cinfo.output_height) {
        int y = cinfo.output_scanline;
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);

        SaveSсanlineToImage(buffer[0], y, result);
    }

    /* Шаг 7: Останавливаем декодирование */

    (void) jpeg_finish_decompress(&cinfo);

    /* Шаг 8: Освобождаем объект декодирования */

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return result;
}

} // of namespace img_lib