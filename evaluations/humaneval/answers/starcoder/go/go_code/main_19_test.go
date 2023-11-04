package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSortNumbers(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("", SortNumbers(""))
    assert.Equal("three", SortNumbers("three"))
    assert.Equal("three five nine", SortNumbers("three five nine"))
    assert.Equal("zero four five seven eight nine", SortNumbers("five zero four seven nine eight"))
    assert.Equal("zero one two three four five six", SortNumbers("six five four three two one zero"))
}
